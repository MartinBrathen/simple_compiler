# llvm-ir code generated for example program([source](https://github.com/MartinBrathen/simple_compiler/blob/master/ebnf.md))

```
declare i32 @f1()

define i32 @f2(i32, i32) {
entry:
  %y = alloca i32
  %x = alloca i32
  store i32 %0, i32* %x
  store i32 %1, i32* %y
  %x1 = load i32, i32* %x
  %y2 = load i32, i32* %y
  %prod = mul i32 %x1, %y2
  ret i32 %prod
}

define i32 @f1.1() {
entry:
  %b = alloca i32
  %a = alloca i32
  %tmp = call i32 @f2(i32 5, i32 3)
  store i32 %tmp, i32* %a
  store i32 0, i32* %b
  br label %lhead

lhead:                                            ; preds = %lbody, %entry
  %whiletmp = phi i32 [ 0, %entry ], [ %sum, %lbody ]
  %b2 = load i32, i32* %b
  %ne = icmp ne i32 %b2, 10
  br i1 %ne, label %lbody, label %cont

lbody:                                            ; preds = %lhead
  %b1 = load i32, i32* %b
  %sum = add i32 %b1, 1
  store i32 %sum, i32* %b
  br label %lhead

cont:                                             ; preds = %lhead
  br i1 true, label %then, label %else

then:                                             ; preds = %cont
  %a3 = load i32, i32* %a
  %sum4 = add i32 %a3, 3
  store i32 %sum4, i32* %a
  br label %ifcont

else:                                             ; preds = %cont
  %a5 = load i32, i32* %a
  %sum6 = add i32 %a5, 5
  store i32 %sum6, i32* %a
  br label %ifcont

ifcont:                                           ; preds = %else, %then
  %iftmp = phi i32 [ %sum4, %then ], [ %sum6, %else ]
  %a7 = load i32, i32* %a
  %b8 = load i32, i32* %b
  %sum9 = add i32 %a7, %b8
  ret i32 %sum9
}
```

# Phi nodes and allocations

A phi node is used after every if statement, to decide which branch's assignments are to be used.

Phi nodes are also used in the while loop's head, because it can be entered from above the loop and also from within its body.

Allocations were used when declaring new variables and functions. all variables are stored in the entry block, even variable declarations inside loops and if statement, which is not ideal.

# Requirements

- Basic code generation.
- Pass noalias where possible allowing for better optimization (assuming your borrowchecker prevents aliasing).
- Other attributes, intrinsics, etc. that enables further LLVM optimizations.

Only basic code generation was done. I borrowed a lot of code and got inspiration from your crust.rs example, and also the inkwell kaleidoscope example.
