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

- Things like `OpenMPLoopConstruct` which have a bunch of things like a begin directive, optional `DoConstruct` and an optional end directive are implemented as `std::tuple`. All parser nodes can be found in `llvm-project/flang/include/flang/Parser/parse-tree.h`

- Most of the classes in `/flang/include/flang/Parser/parse-tree.h` are created as boilerplate macros which makes a lot of sense: common code defined as macro expansion. Most classes simply wrap a structure: `std::tuple` or `std::variant` or popularly the `CharBlock` (dealt with in `/flang/include/Parser/char-block.h`: which basically defines a single alphanumeric character mapped to a parser node to trace back source and other contextual information)

- Best way to check OpenMP semantics is to head over to the file `llvm-project/flang/lib/Semantics/check-omp-structure.cpp`. Let's say you need to semantically validate `OpenMPAtomicConstruct`. Create a suitable class (outside of `OmpStructureChecker`) and within `OmpStructureChecker::Enter(const auto OpenMPAtomicConstruct&)`, initiate a `parser::walk(..., <class here>)` with `<class here>` replaced with the object of the class you created. Then this `<class here>` can have all sorts of `Pre` and `Post` you need for the semantic checks.

### OpenMP

**What is OpenMP?**: OpenMP is an application programming interface that supports multi-platform shared-memory multiprocessing programming in C, C++, and Fortran, on many platforms, instruction-set architectures and operating systems, including Solaris, AIX, HP-UX, Linux, macOS, and Windows.

- `sections` directive: defines independent sections that can be distributed amongst threads. Example

```f90
program sample
    use omp_lib

!$omp sections
    !$omp section
        ! do work here to be given to worker thread 1

    !$omp section
        ! do work here to be given to worker thread 2
!$omp end sections 
end program eample
```

- `simd` directive: defines the things related to SIMD instructions (mainly have non-branching loops within them)

- `atomic` directive: defines a set of statements which must be done **atomically**. Can include a bunch of reads, writes, updates etc. Defined as `OpenMPAtomicConstruct` node in `llvm-project/flang/include/flang/Parser/parse-tree.h`. 

## Patch discussion (verbatim)

Verbatim copy of the important patch discussions to keep everything in one place

### D108904

According to OpenMP 5.0 spec document, the following semantic restrictions have been dealt with in this patch.

- [sections] **Orphaned section directives are prohibited. That is, the section directives must appear within the sections construct and must not be encountered elsewhere in the sections region.**

Semantic checks for the following are not necessary, since use of orphaned section construct (i.e. without an enclosing sections directive) throws parser errors and control flow never reaches the semantic checking phase.
Added a test case for the same in `llvm-project/flang/test/Parser/omp-sections01.f90`

- [sections] **Must be a structured block**

Added test case as `llvm-project/flang/test/Semantics/omp-sections02.f90` and made changes to branching logic in `llvm-project/flang/lib/Semantics/check-directive-structure.h` as follows:

 `NoBranchingEnforce` is used within a `parser::Walk` called from `CheckNoBranching` (in `check-directive-structure.h` itself). `CheckNoBranching` is invoked from a lot of places, but of our interest in this particular context is `void OmpStructureChecker::Enter(const parser::OpenMPSectionsConstruct &x)` (defined within `llvm-project/flang/lib/Semantics/check-omp-structure.cpp`)

 [addition] `NoBranchingEnforce` lacked tracking context earlier (precisely once control flow entered a OpenMP directive, constructs were no longer being recorded on the `ConstructStack`). Relevant `Pre` and `Post` functions added for the same
[addition] A `EmitBranchOutError` on encountering a `CALL` in a structured block

 [changes] Although `NoBranchingEnforce` was handling labelled `EXIT`s and `CYCLE`s well, there were no semantic checks for unlabelled `CYCLE`s and `EXIT`s. Support added for them as well. For unlabeled cycles and exits, there's a need to efficiently differentiate the `ConstructNode` added to the stack before the concerned OpenMP directive was encountered and the nodes added after the directive was encountered. The same is done through a simple private unsigned counter within `NoBranchingEnforce` itself

 [addition] Addition of `EmitUnlabelledBranchOutError` is due to lack of an error message that gave proper context to the programmer. Existing error messages are either named error messages (which aren't valid for unlabeled cycles and exits) or a simple CYCLE/EXIT statement not allowed within SECTIONS construct which is more generic than the reality.


 Other than these implementations, a test case for simd construct has been added.

 - [Test case for simd] **Must be a structured block / A program that branches in or out of a function with declare simd is non conforming.**

Uses the already existing branching out of OpenMP structured block logic as well as changes introduced above to semantically validate the restriction. Test case added to `llvm-project/flang/test/Semantics/omp-simd01.f90`

----------------------

**High priority TODOs**:

1. (done) Decide whether CALL is an invalid branch out of an OpenMP structured block
2. (done) Fix !$omp do's handling of unlabeled CYCLEs