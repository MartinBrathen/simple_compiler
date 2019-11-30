# Ill formed borrow

Following are two example programs where ill formed borrows are demonstrated.

## Example 1
```
fn fun() -> i32 {
  let x: &i32;
  
  if true {
    let y: i32 = 5;
    x = &y; // x is assigned borrowed value from y
  }         // borrowed value goes out of scope
  
  println!("{}",*x); // atempt to use x

  return 0;
}
```
## Example 2
```
fn fun() -> &i32 {
  let x: &i32;
  
  let y: i32 = 5;
  x = &y; // this is now OK because x's lifetime does not extend y's
  
  return &y; // returning a reference to a value that will go out of scope. not good
}
```

# Well formed borrow

This example program fixed the issues that were present in the previous programs.

```
fn fun() -> i32 {
  let x: &i32;
  let y: i32 = 5;
  x = &y; // this is now OK because x's lifetime does not exceed y's
  println!("{}",*x); // x can be used.
  return y; // returning value instead of reference
}
```

# Borrowchecking rules

I didn't implement references/borrow checking, but here are the rules that I would use the check for ill formed borrows:
- Variable from outer scope is assigned borrow from variable in current scope.
this rule is probably stricter than the corresponding rule in rust, since rust seems to allow this when the outer variable
is only used in the current scope before being assigned a new value.
- Function returning reference to a value that is owned by itself.
