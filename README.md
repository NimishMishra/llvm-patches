# llvm-patches

A bunch of things related to my work on LLVM compiler infrastructure

Phabricator profile link: [https://reviews.llvm.org/p/NimishMishra/](https://reviews.llvm.org/p/NimishMishra/)

## Issues

[[flang][OpenMP][omp2012] Type confusion in reduction clause and fortran intrinsic](https://github.com/llvm/llvm-project/issues/67332)

[Crash observed in threadprivatised common block](https://github.com/llvm/llvm-project/issues/65034)

[reduction(min) gives an inexact negative value close to zero](https://github.com/llvm/llvm-project/issues/73101)

## Patches

[[Merged](https://github.com/llvm/llvm-project/commit/d4717b9b9def89c503a20eaf7700f87c4b52d530)][Test case for semantic checks for OpenMP parallel sections contruct](https://reviews.llvm.org/D111438)

[[Merged](https://github.com/llvm/llvm-project/commit/3519dcfec22963fbb84e154cecc2df22e6c7724f)][Semantic checks for OpenMP atomic construct](https://reviews.llvm.org/D110714)

[[Merged](https://github.com/llvm/llvm-project/commit/fe2d053c4505b7ccc8a86e266e68d2f97aaca1e1)][Semantic checks for OpenMP critical construct name resolution](https://reviews.llvm.org/D110502)

[[Merged](https://github.com/llvm/llvm-project/commit/063c5bc31b89d85aba9ea7c2aa0d2440ec468ed2)][Semantic checks for OpenMP constructs: sections and simd](https://reviews.llvm.org/D108904) 

[[Merged](https://github.com/llvm/llvm-project/commit/88d5289fc69d24e8490a064c87228d68c53e5d9c)][Lowering for sections constructs](https://reviews.llvm.org/D122302)

[[Merged](https://github.com/llvm/llvm-project/commit/00c511b35195616430112e5d537588dfdae42cba)][Lowering for atomic read and write constructs](https://reviews.llvm.org/D122725)

[[Merged](https://github.com/flang-compiler/f18-llvm-project/commit/ed5bf452f17805c5cac57433862076cec9469e22)][Cherrypicking atomic operations downstream](https://github.com/flang-compiler/f18-llvm-project/pull/1570)

[[Merged](https://github.com/llvm/llvm-project/commit/435feefbdd6c91faf24fa5e69c4e7c3bc127568a)][Lowering for default clause](https://reviews.llvm.org/D123930)

[[Merged](https://github.com/llvm/llvm-project/commit/6a3c4a40f4ba6943c5902660858d0c3df2830330)][Parser support for in_reduction clause on OpenMP task directive](https://reviews.llvm.org/D124156)

[[Merged](https://github.com/llvm/llvm-project/commit/a56b76d9ca520b4495da2a984ff481a6487736fc)][Lowering for atomic update construct](https://reviews.llvm.org/D125668)

[Additional semantic checks for openmp atomic construct](https://reviews.llvm.org/D126612)

[Lowering for atomic capture construct](https://reviews.llvm.org/D127272)

[[Merged](https://github.com/llvm/llvm-project/commit/7dc18a62e40e241019ec77e70f01bc41d39ab748)][Added semantic checks for hint clause](https://reviews.llvm.org/D127615)

[Added semantic checks for atomic capture, write, and update statements](https://reviews.llvm.org/D127620)

[[Merged](https://github.com/llvm/llvm-project/commit/c05b99971fa042ae75c1848e6597c068392d693b)][Refactor code related to OpenMP atomic memory order clause semantics](https://reviews.llvm.org/D127822)

[Semantic checks for 'operator' in atomic update assignment statements](https://reviews.llvm.org/D128151)

[[Merged](https://github.com/llvm/llvm-project/commit/0ff9625c0a15a80d2375c1ab6fb3fd8439b3c78b)][Semantic checks for symbols in atomic update assignment statement](https://reviews.llvm.org/D128162)

[[Merged](https://github.com/llvm/llvm-project/commit/5cbe39ef880e0d8a61db111fca0b14b3ca17a17b)][Allow default(none) to access variables with PARAMETER attribute](https://reviews.llvm.org/D129444)

[[Merged](https://github.com/llvm/llvm-project/commit/58753dea15c8ff72034e3e4a13b3a945aa295366)][Fix warning due to uninitialized pointer dereference during atomic update lowering](https://reviews.llvm.org/D129914)

[[Merged](https://github.com/llvm/llvm-project/commit/1f734b0d1731bf15fe2501ec04f65658ad4a87a6)][Handle private/firstprivate clauses on sections construct](https://reviews.llvm.org/D131463)

[[Merged](https://reviews.llvm.org/rGf49b6afc231242dfee027d5da69734836097cd43)][Handle lastprivate on sections construct](https://reviews.llvm.org/D133686)

[[Merged](https://github.com/llvm/llvm-project/commit/e5aa6eeb4c474de99c44a667e3852f90e3774396)][Verify support for private/firstprivate on unstructured sections](https://reviews.llvm.org/D145352)

[[Merged](https://github.com/llvm/llvm-project/commit/89ebea8c1e6ff3dfd37943fdc01df409e9bd1e11)][Fixed internal compiler error with atomic update operation verification](https://reviews.llvm.org/D153217)

[[Merged](https://github.com/llvm/llvm-project/commit/2cb6d0c70bff616cce4dbd4cbdffc085175c739f)][Translating if and final clauses for task construct](https://reviews.llvm.org/D130704)

[[Merged](https://github.com/llvm/llvm-project/commit/b4c54b20270ea1f0eac574785156c668930da5c5)][Support common block in OpenMP private clause](https://reviews.llvm.org/D127215)

[[Merged](https://github.com/llvm/llvm-project/commit/f752265231c2d15590a53e45bcc850acf2450dfc)][Support for privatization in common block](https://reviews.llvm.org/D156120)

[[Merged](https://github.com/llvm/llvm-project/commit/da1f1b229246a64ebf385a133c5528a8ce622731)] Prevent extraneous copy in D156120

[[Merged](https://github.com/llvm/llvm-project/commit/91f92e6a6bfa4a1d963234ba70adb5a7957aeb1e) Fix common block missing symbol crash](https://github.com/llvm/llvm-project/pull/67330) in response to issue [Crash observed in threadprivatised common block](https://github.com/llvm/llvm-project/issues/65034)

[[Merged](https://github.com/llvm/llvm-project/commit/c4d73216150fbc9679fe999e70396262cf352d15)] [Skip default privatization for crashing case](https://github.com/llvm/llvm-project/pull/71922)

[[Merged](https://github.com/llvm/llvm-project/commit/956cf0e5de4b816ccba64206ead8bc2d4fd22158)] [Fix min reduction initialization](https://github.com/llvm/llvm-project/pull/73102)

[[Merged](https://github.com/llvm/llvm-project/commit/6023e2476b5cb2fd84dcb74d805ae2e322160111)] [Error out when assumed rank variable in used as selector in SELECT TYPE statement](https://github.com/llvm/llvm-project/pull/74286). Relevant [issue](https://github.com/llvm/llvm-project/issues/74285)

[[Merged](https://github.com/llvm/llvm-project/commit/506ff547924ab81123865964b7d998a5a378330a)] [[flang][Semantics] Threadprivate symbols are ignored in presence of default clause](https://github.com/llvm/llvm-project/pull/78283). Relevant [issue](https://github.com/llvm/llvm-project/issues/78282)

## Additional notes

A bunch of notes in addition to the discussion in the patches themselves

### Generic notes on what works where in the LLVM codebase

- [D108904](https://reviews.llvm.org/D108904) : contains a `std::visit(common::visitors{...})` which can be used as a boilerplate to iterate over `std::variant` anywhere in the codebase.

- [D108904](https://reviews.llvm.org/D108904) : contains boilerplate `Pre` and `Post` functions to manage `SemanticsContext` anywhere in the `Semantics` of the codebase

- [D110714](https://reviews.llvm.org/D110714): contains a reusable `HasMember` way of recognising whether the `typename` in a templated function belongs to a certain variant. This could be treated as a complementary way to `std::visit`.

- [D127047](https://reviews.llvm.org/D127047): contains ideas on greating global variables (`embox` ,`undefined`, and `has_value`). And a good example of using `std::function` as passing arguments to a function.

### Fortran specific features

- Printing an unassociated pointer or unallocated allocatable may cause one segfault in some cases if someone tries to test execution. So, add one external function call to handle unassociated pointer and unallocated allocatable variables.

### Parser

- Possibly the most important source code resides in `llvm-project/flang/include/flang/Parser/parse-tree.h` which contains all the parse tree nodes' definition.

- The generic structure is the same for all. Suppose OpenMP critical construct needs to be defined. Therefore first the individual components will be defined: `struct OmpCriticalDirective`, `struct Block`, and `struct OmpEndCriticalDirective` and then `struct OpenMPCriticalConstruct` shall be defined as a wrapper for a tuple `std::tuple<OmpCriticalDirective, Block, OmpEndCriticalDirective>`.

- Most classes share code. Therefore you will find a lot of macros as class boilerplates. Boilerplates usually wrap very simple functionality and functions.

- Most of the parsing activities are done by `TYPE_PARSE` that is defined in `llvm-project/flang/lib/Parser/type-parser-implementation.h`, which is a wrapper around `parser.Parse(state)` where `state` is `ParserState`. The actual `struct Parser` template is defined in `llvm-project/flang/lib/Parser/type-parsers.h` which is the base for all other parser instantiations.

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
end program sample
```

- `simd` directive: defines the things related to SIMD instructions (mainly have non-branching loops within them)

- `atomic` directive: defines a set of statements which must be done **atomically**. Can include a bunch of reads, writes, updates etc. Defined as `OpenMPAtomicConstruct` node in `llvm-project/flang/include/flang/Parser/parse-tree.h`. 

- `critical`: defines a section of code as critical, implies only one thread in a thread worker group shall enter into the section at a time

OpenMP is a bunch of pragmas you can put in your code to tell the compiler how to handle them exactly. For example, `!$omp atomic read` above a `x = y` means the compiler should perform this atomically. OpenMP is a **dialect**. And it is upto compiler engineers to build support in both flang and mlir for OpenMP dialect.

- OpenMP standard (updated till standard 5.1) has `hint` expressions attached with OpenMP constructs. These are mostly synchronization hints that you use provide to the compiler in order to optimize critical sections (thereby two most popular constructs here are `atomic` and `critical`). Few main types of hint expressions:
    
    -- **uncontended**: expect low contention. Means expect few threads will contend for this particular critical section

    -- **contended**: expect high contention.
    
    -- **speculative**: use speculative techniques like transactional memory. Just like DBMS involves transactions as a group (they are committed and rolled back as a group , similar ideas are ported to concurrent executions of critical sections

    -- **non-speculative**: do not use speculative techniques as transactional memory

- Getting expressions is one of the core things you have to perform a lot. The `GetExpr` interface is defined in `flang/include/flang/Semantics/tools.h`. In MLIR, use AbstractConverter's `genExprValue`; you may have to work with `fir::ExtendableValue` (which is further defined in `flang/Optimizer/Builder/BoxValue.h`). From here, you can extract `mlir::Value` through a `fir::getBase` (this gets the final value [the temporary variable number] from the computation of the entire expression.

### MLIR

- An infrastructure where you can define a bunch of things aiding your compiler infrastructure needs. You can define your own operations, type systems etc and reuse MLIR's pass management, threading etc to get a compiler infrastructure up and running really quick. For example, we import MLIR functionality into flang, lower PFT to MLIR, and let MLIR do its magic, and finally lower to LLVM. This integration is seemless.

- MLIR can be used to design custom IR for a variety of use-cases. For example, the same MLIR infra can be used to generate code for Fortran as well as Tensorflow.

- Clang builds some AST and performs transformations on the same. This is fairly successful. But people have started looking at how retaining some of the high level semantics can help in better transformations. LLVM IR loses this context; thus MLIR hopes to retain such high level semantic information and perform transformations on the same.

- **Dominance checking**: A part of code resides in `mlir/lib/IR/Verifier.cpp` which manipulates the regions of different things, understands what region envelops what region, and so on.

#### FIR

- Fortran has the concept of arrays whose lower bound, upper bound, and sizes are defined at runtime. This means we have to keep this information somewhere in memory. And the best place to do that is the array itself. Fortran array descriptors abstract these informations (how many dimensions, what the stride from one element to the next. It also contains type information for the type itself). In FIR dialect, the `fir.box` is used to say **this is a descriptor** and `fir.embox` is used to say **package this as a descriptor**.

### PFT to MLIR lowering

MORE INFORMATION WILL BE ADDED AS TIME PASSES

- You need to have a bunch of definable operations which you can create lowering code for. In namespace `mlir` in `llvm-project/mlir/lib/Conversion/PassDetails.h`, there is a child namespace called `namespace omp` that encapsulates `class OpenMPDialect` which in turn is actually in an inc file you will find in your `build` and `install` folder. `build`: `tools/mlir/include/mlir/Dialect/OpenMP/OpenMPOpsDialect.h.inc`; similarly for `install` folder too. 

- To include a new operation, do it in `llvm-project/mlir/include/mlir/Dialect/OpenMP/OpenMPOps.td`. The corresponding created class will be in `install/include/mlir/Dialect/OpenMP/OpenMPOps.h.inc` and all function definitions will be in `install/include/mlir/Dialect/OpenMP/OpenMPOps.cpp.inc`. You can define your custom builders etc in the `llvm-project/mlir/include/mlir/Dialect/OpenMP/OpenMPOps.td` file, whose definition goes in `llvm-project/mlir/lib/Dialect/OpenMP/IR/OpenMPDialect.cpp`.

- For `Variadic<AnyType>:$private_vars` and `Variadic<AnyType>:$firstprivate_vars`, you need to extract the clause list, extract these clauses, and convert their arguments to an object list through `genObjectList` that fits in everything in `SmallVector<Value>`. `genObjectList` basically extracts the symbol of every argument and gets an address reference for the same. This is later used to create FIR.

- Most of the OMP clauses are defined in `llvm-project/llvm/include/llvm/Frontend/OpenMP/OMP.td`. For example

```td
def OMPC_Reduction : Clause<"reduction">{
    let clangClass = "OMPReductionClause";
    let flangClass = "OmpReductionClause";
}

def OMP_Sections : Directive<"sections">{
    let allowedClauses = [
        VersionedClause<OMPC_Private>,
        VersionedClause<OMPC_LastPrivate>,
        VersionedClause<OMPC_FirstPrivate>,
        VersionedClause<OMPC_Reduction>,
        VersionedClause<OMPC_NoWait>,
        VersionedClause<OMPC_Allocate>
    ];
}
```

basically define OMP classes for both clang and flang, and a bunch of pragmas or directives to be used in both places as well as the allowed clauses in the clause list.

- To know how exactly the clauses must be handled, refer to `llvm-project/llvm/include/llvm/Frontend/OpenMP/OMP.td`, pick one clause like `OMPC_Reduction`, and check its `clangClass` and `flangClass` entry. This `flangClass`'s details will be found in `llvm-project/flang/include/flang/Parser/parse-tree.h`.

- MLIR is a general purpose intermediate representation. It does not understand external types. In order to do so, you need to do a `public mlir::omp::PointerLikeType::ExternalModel` wherein you attach an external type to the pointer model. This is just a wrapper around a `getElementType` method that *casts* the `mlir::Type pointer` to the type that MLIR wants to capture from the external source.

- In `llvm/include/llvm/Frontend/OpenMP/OMP.td`, OMPC (OMP clauses) are defined as given here. In order to lower these, MLIR usually expects a `String::Attr`, wherein you need to convert the enum clause value to `mlir::StringAttr`. In order to do so, MLIR provides an interface `stringifyClause + <NAME>` (like `stringifyClauseMemoryOrderKind`) which takes in an argument of the form `omp::ClauseMemoryOrderKind::acquire`. This will end up stringifying the entire thing.

```td
def OMPC_MemoryOrder : Clause<"memory_order"> {
    let enumClauseValue = "MemoryOrderKind";
    let allowedClauseValues = [
        OMP_MEMORY_ORDER_SeqCst,
        OMP_MEMORY_ORDER_AcqRel,
        OMP_MEMORY_ORDER_Acquire,
        OMP_MEMORY_ORDER_Release,
        OMP_MEMORY_ORDER_Relaxed,
        OMP_MEMORY_ORDER_Default
    ];
}    
```

## Patch discussion (verbatim)

Verbatim copy of the important patch discussions to keep everything in one place

### D110714

The following patch implements the following semantic checks for atomic construct based on OpenMP 5.0 specifications.

- **In atomic update statement, binary operator is one of +, *, -, /, .AND., .OR., .EQV., or .NEQV.**

In `llvm-project/flang/lib/Semantics/check-omp-structure.cpp`, a new class `OmpAtomicConstructChecker` is added for all semantic checks implemented in this patch. This class is used in a `parser::Walk` in `OmpStructureChecker::CheckOmpAtomicConstructStructure` in `check-omp-structure.cpp` itself.

The entire aim is to, for any OpenMP atomic update statement, initiate a `parser::Walk` on the construct, capture the assignment statement in `bool Pre(const parser::AssignmentStmt &assignment)` inside `OmpAtomicConstructChecker`, and initiate a check with `CheckOperatorValidity` for the operator validity.

`CheckOperatorValidity` has two `HasMember` checks. The first check is to see whether the assignment statement has a binary operator or not. Because if it doesn't, the call to `CompareNames` is unsuccessful. The second check is to see if the operator is an allowed binary operator.

- **In atomic update statement, the statements must be of the form x = x operator expr or x = expr operator x**

The `CompareNames` function called from within `CheckOperatorValidity` checks that if we have an assignment statement with a binary operator, whether the variable names are same on the LHS and RHS.

- **In atomic update statement, only the following intrinsic procedures are allowed: MAX, MIN, IAND, IOR, or IEOR**

The functionality for the same is added in the function `CheckProcedureDesignatorValidity`.

--------------
Alternative approaches tried which we could discuss upon:

- In `CheckOperatorValidity`, I earlier tried a `std::visit` approach. But that required us to enumerate all **dis**-allowed binary operators. Because had we emitted errors for anything that is not allowed, we began facing issues with assignment statements in atomic READ statements.

- In `llvm-project/flang/include/flang/Parser/parse-tree.h`, the `struct Expr` has a variant where all relevant operators and procedure indirection things are present. One interesting structure is `IntrinsicBinary` which is the base structure for all relevant binary operators in `parser::Expr`. However, we found no way to capture the different binary operators through this base structure. Concretely, I was thinking something like this:

```c
std::visit(common::visitors{

[&](const parser::Expr::IntrinsicBinary &x){
        // check for validity of variable names as well as operator validity   
    },
[&](const auto &x){
        // got something other than a valid binary operator. Let it pass. 
    },
}, node);
```

### D110502

Taken forward from https://reviews.llvm.org/D93051 (authored by @sameeranjoshi )

As reported in https://bugs.llvm.org/show_bug.cgi?id=48145, name resolution for omp critical construct was failing. This patch adds functionality to help that name resolution as well as implementation to catch name mismatches.

- **Changes to check-omp-structure.cpp**

 In `llvm-project/flang/lib/Semantics/check-omp-structure.cpp`, under `void OmpStructureChecker::Enter(const parser::OpenMPCriticalConstruct &x)`, logic is added to handle the different forms of name mismatches and report appropriate error. The following semantic restrictions are therefore handled here:

    - If a name is specified on a critical directive, the same name must also be specified on the end critical directive

    - If no name appears on the critical directive, no name can appear on the end critical directive

    - If a name appears on either the start critical directive or the end critical directive

The only allowed combinations are: (1) both start and end directives have same name, (2) both start and end directives have NO name altogether

- **Changes to resolve-directives.cpp**

 In `llvm-project/flang/lib/Semantics/resolve-directives.cpp`, two `Pre` functions are added for `OmpCriticalDirective` and `OmpEndCriticalDirective` that invoke the services of `ResolveOmpName` to give new symbols to the names. This aids in removing the undesirable behavior noted in https://bugs.llvm.org/show_bug.cgi?id=48145

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
