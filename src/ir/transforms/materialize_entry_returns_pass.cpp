/*
 * Copyright (c) PyPTO Contributors.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * -----------------------------------------------------------------------------------------------------------
 */
/**
 * @file materialize_entry_returns_pass.cpp
 * @brief Give a dispatch entry's return values a host-visible output buffer.
 *
 * The only output channel the runtime offers an artifact is a caller-supplied
 * tensor argument: ``ConvertTensorToTileOps`` appends an Out param to each
 * InCore callee for its returns and allocates the matching buffer at the call
 * site, but for the *entry* Orchestration function that call-site buffer is an
 * internal ``tensor.create`` — orchestration codegen writes the task output
 * into a scope-local allocation and nothing exports it, so a return-style
 * kernel (outputs as ``return`` values, no ``pl.Out`` parameters) compiled,
 * executed, and handed the host ``()``.
 *
 * This pass closes that gap at the signature level, so every downstream
 * consumer keeps its existing in-place machinery: each tensor return of a
 * dispatch entry that does not already write back an Out/InOut param is traced
 * to its backing buffer; when that buffer is a fresh top-level
 * ``tensor.create``, the create is deleted and the buffer is appended to the
 * entry's params as an Out param. Codegen then binds it to
 * ``orch_args.tensor(<n>)`` like any declared output, and the artifact's
 * dispatch metadata (derived from the post-pass program) reports it as an
 * auto-allocatable output slot.
 *
 * A "dispatch entry" is an Orchestration function no other function in the
 * program calls — the root of the call graph. Multi-orch (L2) programs have
 * several, each dispatched through its own sub-build; an Orchestration
 * function called by another function (a graph fragment, a chip task under an
 * L3 host orchestrator) is a callee and is left alone, because appending a
 * param would break its call sites' full-coverage obligation.
 *
 * Returns that do not trace to a promotable buffer are left untouched, by
 * design: the trace stops at views (a reshaped alias of an Out param — the
 * data is already host-visible through the param it writes), at extern-kernel
 * call results (extern members carry no IR body, so lineage cannot cross
 * them, but the output flows through the declared Out arg), and at
 * orchestration-level computed values (``tensor.add`` and friends — programs
 * that cannot be code-generated anyway, since compute ops are only registered
 * inside InCore blocks). Rejecting those loudly would break in-place-style
 * programs whose return expressions are views; the pre-fix behavior (the
 * return value carries no extra delivery obligation) is the correct one for
 * all three.
 *
 * Runs after ``ConvertTensorToTileOps`` (the internal creates exist) and
 * before ``NormalizeReturnOrder`` (whose return canonicalization then sees the
 * final param set).
 */

#include <cstddef>
#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "pypto/core/logging.h"
#include "pypto/ir/expr.h"
#include "pypto/ir/function.h"
#include "pypto/ir/kind_traits.h"
#include "pypto/ir/op_registry.h"
#include "pypto/ir/program.h"
#include "pypto/ir/span.h"
#include "pypto/ir/stmt.h"
#include "pypto/ir/transforms/base/mutator.h"
#include "pypto/ir/transforms/base/visitor.h"
#include "pypto/ir/transforms/pass_properties.h"
#include "pypto/ir/transforms/passes.h"
#include "pypto/ir/transforms/utils/auto_name_utils.h"
#include "pypto/ir/transforms/utils/return_lineage_utils.h"
#include "pypto/ir/type.h"

namespace pypto {
namespace ir {
namespace {

/// Collect the names of every function referenced by a Call/Submit in a body.
class CalleeNameCollector : public IRVisitor {
 public:
  std::unordered_set<std::string> callee_names;

 protected:
  void VisitExpr_(const CallPtr& call) override {
    Record(call);
    IRVisitor::VisitExpr_(call);
  }

  /// A task launch forwards its callee exactly as a plain call does, and the
  /// base visitor does not route ``Submit`` through the ``Call`` handler
  /// (`.claude/rules/pass-submit-awareness.md`).
  void VisitExpr_(const SubmitPtr& submit) override {
    Record(SubmitToCallView(submit));
    IRVisitor::VisitExpr_(submit);
  }

 private:
  void Record(const CallPtr& call) {
    if (auto global_var = std::dynamic_pointer_cast<const GlobalVar>(call->op_)) {
      callee_names.insert(global_var->name_);
    }
  }
};

[[nodiscard]] bool IsFreshTensorCreate(const AssignStmtPtr& assign) {
  auto call = As<Call>(assign->value_);
  return call != nullptr && IsOp(call, "tensor.create");
}

/// Rewrite every expression reference of a promoted create var onto its new
/// param var. Positional call binding is by Var identity, so replacing the arg
/// var with the param var itself is what binds the call's Out slot to the
/// host-supplied buffer.
class PromotedVarRewriter : public IRMutator {
 public:
  explicit PromotedVarRewriter(std::unordered_map<const Var*, VarPtr> replacements)
      : replacements_(std::move(replacements)) {}

 protected:
  ExprPtr VisitExpr_(const VarPtr& op) override {
    auto it = replacements_.find(op.get());
    if (it == replacements_.end()) return op;
    return it->second;
  }

 private:
  std::unordered_map<const Var*, VarPtr> replacements_;
};

/// Promote a dispatch entry's return-style outputs to appended Out params.
/// Returns nullptr when the entry needs no change.
FunctionPtr MaterializeEntry(const FunctionPtr& func, const ProgramPtr& program) {
  auto return_stmt = return_lineage::FindFirstReturn(func->body_);
  if (!return_stmt || return_stmt->value_.empty()) return nullptr;

  auto seq = As<SeqStmts>(func->body_);
  if (!seq) return nullptr;

  // Return positions already bound to a param (the in-place spelling) keep
  // their declared buffer; only the unbound ones need materialization.
  const auto param_map = return_lineage::ReturnedParamIndices(func, program);

  // The promotable buffers are fresh top-level creates. A create nested in a
  // loop or scope has per-iteration lifetime semantics a host buffer cannot
  // reproduce, so those are rejected below rather than silently mis-promoted.
  std::unordered_map<const Var*, AssignStmtPtr> top_level_creates;
  for (const auto& stmt : seq->stmts_) {
    if (auto assign = As<AssignStmt>(stmt); assign && assign->var_ && IsFreshTensorCreate(assign)) {
      top_level_creates.emplace(assign->var_.get(), assign);
    }
  }

  std::unordered_set<std::string> taken_base_names;
  for (const auto& param : func->params_) {
    taken_base_names.insert(auto_name::GetCompatibleBaseName(param->name_hint_));
  }

  std::vector<VarPtr> new_params = func->params_;
  std::vector<ParamDirection> new_directions = func->param_directions_;
  std::unordered_map<const Var*, VarPtr> promoted;
  std::unordered_set<const AssignStmt*> removed_creates;
  std::vector<ExprPtr> new_return_values = return_stmt->value_;
  size_t out_counter = 0;
  bool changed = false;

  for (size_t i = 0; i < return_stmt->value_.size(); ++i) {
    if (i < param_map.size() && param_map[i].has_value()) continue;  // in-place writeback
    auto var = AsVarLike(return_stmt->value_[i]);
    if (!var || !AsTensorTypeLike(var->GetType())) continue;  // non-tensor returns are out of scope
    auto buffer = return_lineage::TraceToBackingBuffer(var, func->body_, func->params_, program);
    if (!buffer.has_value()) {
      // Untraceable: a view of an Out param, an extern-call result, or a
      // graph-level computed value. None of these carries a fresh buffer to
      // promote, and the first two deliver through their declared Out params
      // already — leave the return as-is (see the file comment).
      continue;
    }
    if (buffer->is_param) continue;  // already host-supplied

    auto create_it = top_level_creates.find(buffer->var.get());
    if (create_it == top_level_creates.end()) {
      // The backing buffer is a create nested in a loop or scope (per-iteration
      // lifetime a host buffer cannot reproduce). Leave it internal — the
      // return keeps its pre-fix non-delivery rather than being mis-promoted.
      continue;
    }

    auto promoted_it = promoted.find(buffer->var.get());
    if (promoted_it == promoted.end()) {
      // Collision-free naming: two calls to the same callee in one entry each
      // synthesize a create named `ret0__out`, so the counter walks past any
      // base name already taken by a declared or promoted param.
      std::string name = auto_name::BuildName("ret" + std::to_string(out_counter++), "", "out");
      while (taken_base_names.count(auto_name::GetCompatibleBaseName(name)) != 0) {
        name = auto_name::BuildName("ret" + std::to_string(out_counter++), "", "out");
      }
      taken_base_names.insert(auto_name::GetCompatibleBaseName(name));
      auto param_var = std::make_shared<Var>(name, buffer->var->GetType(), buffer->var->span_);
      promoted.emplace(buffer->var.get(), param_var);
      new_params.push_back(param_var);
      new_directions.push_back(ParamDirection::Out);
      removed_creates.insert(create_it->second.get());
    }
    new_return_values[i] = promoted.at(buffer->var.get());
    changed = true;
  }
  if (!changed) return nullptr;

  std::vector<StmtPtr> new_stmts;
  new_stmts.reserve(seq->stmts_.size());
  bool return_rewritten = false;
  for (const auto& stmt : seq->stmts_) {
    if (auto assign = As<AssignStmt>(stmt); assign && removed_creates.count(assign.get()) != 0) {
      continue;
    }
    if (stmt.get() == return_stmt.get()) {
      // CtrlFlowTransform makes returns function-terminal, so the entry's
      // ReturnStmt is a top-level child; rebuild it with the promoted params.
      new_stmts.push_back(std::make_shared<ReturnStmt>(new_return_values, return_stmt->span_));
      return_rewritten = true;
      continue;
    }
    new_stmts.push_back(stmt);
  }
  INTERNAL_CHECK_SPAN(return_rewritten, func->span_)
      << "Internal error: dispatch entry '" << func->name_
      << "' has a ReturnStmt that is not a top-level statement of its body";

  PromotedVarRewriter rewriter(promoted);
  auto new_body = rewriter.VisitStmt(std::make_shared<SeqStmts>(std::move(new_stmts), seq->span_));

  return std::make_shared<Function>(func->name_, std::move(new_params), std::move(new_directions),
                                    func->return_types_, std::move(new_body), func->span_, func->func_type_,
                                    func->level_, func->role_, func->attrs_);
}

}  // namespace

namespace pass {

Pass MaterializeEntryReturns() {
  auto pass_func = [](const ProgramPtr& program) -> ProgramPtr {
    // A dispatch entry is an Orchestration function nothing else calls: the
    // host invokes it directly, so it owns an artifact calling convention.
    std::unordered_set<std::string> callee_names;
    for (const auto& [gvar, func] : program->functions_) {
      if (!func || !func->body_) continue;
      CalleeNameCollector collector;
      collector.VisitStmt(func->body_);
      callee_names.insert(collector.callee_names.begin(), collector.callee_names.end());
    }

    std::vector<FunctionPtr> functions;
    bool modified = false;
    for (const auto& [gvar, func] : program->functions_) {
      if (func && func->func_type_ == FunctionType::Orchestration && callee_names.count(func->name_) == 0) {
        if (auto materialized = MaterializeEntry(func, program)) {
          functions.push_back(std::move(materialized));
          modified = true;
          continue;
        }
      }
      functions.push_back(func);
    }
    if (!modified) return program;
    return std::make_shared<Program>(std::move(functions), program->name_, program->span_);
  };

  return CreateProgramPass(pass_func, "MaterializeEntryReturns", kMaterializeEntryReturnsProperties);
}

}  // namespace pass
}  // namespace ir
}  // namespace pypto
