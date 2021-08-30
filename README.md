# llvm-patches

A bunch of things related to my work on LLVM compiler infrastructure

## Patches

[Semantic checks for OpenMP constructs: sections and simd](https://reviews.llvm.org/D108904)

## Additional notes

A bunch of notes in addition to the discussion in the patches themselves

### Generic notes on what works where in the LLVM codebase

- [D108904](https://reviews.llvm.org/D108904) : contains a `std::visit(common::visitors(...))` which can be used as a boilerplate to iterate over `std::variant` anywhere in the codebase.

- [D108904](https://reviews.llvm.org/D108904) : contains boilerplate `Pre` and `Post` functions to manage `SemanticsContext` anywhere in the `Semantics` of the codebase

### Semantics

- A `std::variant` of `ConstructNode` is maintained as the `ConstructStack` that pushes and pops contextual information (especially when a certain parser node is encountered while navigating the parse tree). This is very useful to look for nesting behaviour or to see if something is properly enclosed within a required construct. Examples of `ConstructNode` include `DoConstruct`, `IfConstruct` etc. However, as of August 30, 2021 (while working on patch [D108904](https://reviews.llvm.org/D108904)), `OpenMPConstruct` is not a part of the stack. It is maintained as a separate `SemanticsContext`

- `parser::Walk` can be used as an iterator over the AST nodes. One argument taken by `parser::Walk` is a certain template class which encapsulates functionality on what to do when certain nodes are encountered: see for `Pre` and `Post` examples in `NoBranchingEnforce` in `flang/lib/Semantics/check-directive-structure.h` 

## Patch discussion (verbatim)

Verbatim copy of the important patch discussions to keep everything in one place

### D108904

According to OpenMP 5.0 spec document, the following semantic restrictions have been dealt with in this patch.

- [sections] **Orphaned section directives are prohibited. That is, the section directives must appear within the sections construct and must not be encountered elsewhere in the sections region.**

Semantic checks for the following are not necessary, since use of orphaned section construct (i.e. without an enclosing sections directive) throws parser errors and control flow never reaches the semantic checking phase.
Added a test case for the same in `llvm-project/flang/test/Parser/omp-sections01.f90`

- [simd] **Must be a structured block / A program that branches in or out of a function with declare simd is non conforming.**

Uses the already existing branching out of OpenMP structured block logic as well as changes introduced in point 3 below to semantically validate the restriction. Test case added to `llvm-project/flang/test/Semantics/omp-simd01.f90`

- [sections] **Must be a structured block**

Added test case as `llvm-project/flang/test/Semantics/omp-sections02.f90` and made changes to branching logic in `llvm-project/flang/lib/Semantics/check-directive-structure.h` as follows:

 `NoBranchingEnforce` is used within a `parser::Walk` called from `CheckNoBranching` (in `check-directive-structure.h` itself). `CheckNoBranching` is invoked from a lot of places, but of our interest in this particular context is `void OmpStructureChecker::Enter(const parser::OpenMPSectionsConstruct &x)` (defined within `llvm-project/flang/lib/Semantics/check-omp-structure.cpp`)

 [addition] `NoBranchingEnforce` lacked tracking context earlier (precisely once control flow entered a OpenMP directive, constructs were no longer being recorded on the `ConstructStack`). Relevant `Pre` and `Post` functions added for the same
[addition] A `EmitBranchOutError` on encountering a `CALL` in a structured block

 [changes] Although `NoBranchingEnforce` was handling labelled `EXIT`s and `CYCLE`s well, there were no semantic checks for unlabelled `CYCLE`s and `EXIT`s. Support added for them as well. For unlabeled cycles and exits, there's a need to efficiently differentiate the `ConstructNode` added to the stack before the concerned OpenMP directive was encountered and the nodes added after the directive was encountered. The same is done through a simple private unsigned counter within `NoBranchingEnforce` itself

 [addition] Addition of `EmitUnlabelledBranchOutError` is due to lack of an error message that gave proper context to the programmer. Existing error messages are either named error messages (which aren't valid for unlabeled cycles and exits) or a simple CYCLE/EXIT statement not allowed within SECTIONS construct which is more generic than the reality.