#![allow(unused_imports)]
#![allow(dead_code)]
extern crate nom;

use inkwell::{
    basic_block::BasicBlock,
    builder::Builder,
    context::Context,
    execution_engine::{ExecutionEngine, JitFunction},
    module::Module,
    passes::PassManager,
    types::{BasicTypeEnum, AnyTypeEnum, FunctionType, IntType},
    values::{BasicValueEnum, FloatValue, FunctionValue, InstructionValue, IntValue, PointerValue},
    FloatPredicate, OptimizationLevel, IntPredicate,

};

use std::error::Error;

use nom::{
    branch::alt,
    bytes::complete::{tag,is_not},
    character::complete::{digit1, multispace0, multispace1,  alpha1, alphanumeric0},
    combinator::{map},
    sequence::{preceded, tuple, terminated},
    IResult,
};

use std::collections::HashMap;
use nom_locate::LocatedSpan;

type Span<'a> = LocatedSpan<&'a str>;

#[derive(Debug, PartialEq)]
pub enum Op {
    Add,
    Mul,
    Div,
    Mod,
}

type SpanOp<'a> = (Span<'a>, Op);

#[derive(Debug, PartialEq)]
pub enum UOp {
    Neg
}

type SpanUOp<'a> = (Span<'a>, UOp);


#[derive(Debug, PartialEq)]
pub enum Expr<'a> {
    Num(i32),
    BinOp(Box<SpanExpr<'a>>, SpanOp<'a>, Box<SpanExpr<'a>>),
    UOp(SpanUOp<'a>, Box<SpanExpr<'a>>),
    Val(bool),
    BinBOp(Box<SpanExpr<'a>>, SpanBOp<'a>, Box<SpanExpr<'a>>),
    UBOp(SpanUBOp<'a>, Box<SpanExpr<'a>>),
    Comp(Box<SpanExpr<'a>>, SpanComp<'a>, Box<SpanExpr<'a>>),
    VarRef(String),
    FCall(String, Box<SpanStatement<'a>>),
}

type SpanExpr<'a> = (Span<'a>, Expr<'a>);


#[derive(Debug, PartialEq)]
pub enum BOp {
    And,
    Or,
}

type SpanBOp<'a> = (Span<'a>, BOp);

#[derive(Debug, PartialEq)]
pub enum UBOp {
    Not,
}

type SpanUBOp<'a> = (Span<'a>, UBOp);

#[derive(Debug, PartialEq)]
pub enum Comp {
    Equal,
    NotEqual,
}

type SpanComp<'a> = (Span<'a>, Comp);



fn parse_add(i: Span) -> IResult<Span, SpanOp> {
    preceded(multispace0,
        map(tag("+"), |s| (s, Op::Add))
    )(i)
}

fn parse_mdm(i: Span) -> IResult<Span, SpanOp> {
    preceded(multispace0,
        alt((
            map(tag("*"), |s| (s, Op::Mul)),
            map(tag("/"), |s| (s, Op::Div)),
            map(tag("%"), |s| (s, Op::Mod)),
        ))
    )(i)
}

fn parse_op(i: Span) -> IResult<Span, SpanOp> {
    preceded(multispace0,
        alt((
            map(tag("*"), |s| (s, Op::Mul)),
            map(tag("/"), |s| (s, Op::Div)),
            map(tag("%"), |s| (s, Op::Mod)),
            map(tag("+"), |s| (s, Op::Add)),
        ))
    )(i)
}

fn parse_neg(i: Span) -> IResult<Span, SpanUOp> {
    preceded(multispace0,
        map(tag("-"), |s| (s, UOp::Neg))
    )(i)
}

fn parse_bop(i: Span) -> IResult<Span, SpanBOp> {
    preceded(multispace0,
        alt((
            map(tag("&&"), |s| (s, BOp::And)),
            map(tag("||"), |s| (s, BOp::Or)),
        ))
    )(i)
}

fn parse_and(i: Span) -> IResult<Span, SpanBOp> {
    preceded(multispace0,
        map(tag("&&"), |s| (s, BOp::And))
    )(i)
}

fn parse_or(i: Span) -> IResult<Span, SpanBOp> {
    preceded(multispace0,
        map(tag("||"), |s| (s, BOp::Or))
    )(i)
}

fn parse_not(i: Span) -> IResult<Span, SpanUBOp> {
    preceded(multispace0,
        map(tag("!"), |s| (s, UBOp::Not))
    )(i)
}

fn parse_comp(i: Span) -> IResult<Span, SpanComp> {
    preceded(multispace0,
        alt((
            map(tag("=="), |s| (s, Comp::Equal)),
            map(tag("!="), |s| (s, Comp::NotEqual)),
        ))
    )(i)
}


fn parse_bool(i: Span) -> IResult<Span, SpanExpr> {
    preceded(multispace0,
        map(alt((tag("true"), tag("false"))), |bool_str: Span| {
            (
                bool_str,
                Expr::Val(bool_str.fragment.parse::<bool>().unwrap()),
            )
        })
    )(i)
}
// Parses Span/string into i32
fn parse_i32(i: Span) -> IResult<Span, SpanExpr> {
    preceded(multispace0,
        map(digit1, |digit_str: Span| {
            (
                digit_str,
                Expr::Num(digit_str.fragment.parse::<i32>().unwrap()),
            )
        })
    )(i)
}

fn parse_var_fcall(i: Span) -> IResult<Span, SpanExpr> {
    alt ((
        
        parse_f_call,
        map(
            parse_var,
            |var| (i,Expr::VarRef(var))
        ),
    ))(i)
}


fn parse_expr_arith(i: Span) -> IResult<Span, SpanExpr> {

    preceded(multispace0,
        alt((

            map( // Parses i32 + expr
                tuple((parse_i32_or_var, parse_add, parse_expr_arith)),
                |(l, op, r)| (i, Expr::BinOp(Box::new(l), op, Box::new(r))),
            ),
            map( // Parses unit - unit (bin op) expr
                tuple((parse_expr_minus, parse_op, parse_expr_arith)),
                |(l, op, r)| (i, Expr::BinOp(Box::new(l), op, Box::new(r))),
            ),
            map( // Parses unit - unit - expr
                tuple((parse_expr_minus, parse_expr_arith)),
                |(l, r)| (i, Expr::BinOp(Box::new(l), parse_add(Span::new("+")).unwrap().1, Box::new(r))),
            ),
            parse_expr_minus,
            map( // Parses unit (bin op) expr
                tuple((parse_expr_mdm, parse_op, parse_expr_arith)),
                |(l, op, r)| (i, Expr::BinOp(Box::new(l), op, Box::new(r))),
            ),
            parse_expr_mdm,
            
        ))
    )(i)

}

// Parses i32 - unit. res: i32 + -unit
fn parse_expr_minus(i: Span) -> IResult<Span, SpanExpr>{
    map( 
        tuple((parse_expr_mdm, parse_expr_arith)),
        |(l, r)| (i, Expr::BinOp(Box::new(l), parse_add(Span::new("+")).unwrap().1, Box::new(r))),
    )(i)
}

// Parses units. ex: i32, (i32+i32), (...)*i32, i32*(...), (...)*(...) etc
fn parse_expr_mdm(i: Span) -> IResult<Span, SpanExpr>{
    preceded(multispace0,
        alt((
            
            map( // Parses i32 (*, /, %) unit
                tuple((parse_i32_or_var, parse_mdm, parse_expr_mdm)),
                |(l, op, r)| (i, Expr::BinOp(Box::new(l), op, Box::new(r))),
            ),
            map( // (expr) (*, /, %) unit
                tuple((preceded(tag("("), parse_expr_arith), preceded(tag(")"),parse_mdm), parse_expr_mdm)),
                |(l, op, r)| (i, Expr::BinOp(Box::new(l), op, Box::new(r))),
            ),
            map( // Parses (expr) (bin op) expr
                tuple((preceded(tag("("), parse_expr_arith), preceded(tag(")"),parse_op), parse_expr_arith)),
                |(l, op, r)| (i, Expr::BinOp(Box::new(l), op, Box::new(r))),
            ),
                 // Parses (expr)
            parse_expr_parentheses,
            map( // Parses - unit
                tuple((parse_neg, alt((parse_i32_or_var,parse_expr_mdm)))),
                |(op, r)| (i, Expr::UOp( op, Box::new(r))),
            ),
            // Parses string to i32
            parse_i32_or_var
        ))
    )(i)
}

fn parse_i32_or_var(i: Span) -> IResult<Span, SpanExpr>{
    preceded(multispace0,
        alt((
            parse_i32,
            parse_var_fcall,
        ))
    )(i)
}



fn parse_expr_bool(i: Span) -> IResult<Span, SpanExpr> {

    preceded(multispace0,
        alt((

            map(
                tuple((parse_bool, parse_or, parse_expr_bool)),
                |(l, op, r)| (i, Expr::BinBOp(Box::new(l), op, Box::new(r))),
            ),
            map(
                tuple((alt((parse_expr_bu, parse_var_fcall)), parse_bop, parse_expr_bool)),
                |(l, op, r)| (i, Expr::BinBOp(Box::new(l), op, Box::new(r))),
            ),

            parse_expr_bu,
            
        ))
    )(i)

}


fn parse_expr_bu(i: Span) -> IResult<Span, SpanExpr>{
    preceded(multispace0,
        alt((
            map( // Parses i32 (*, /, %) unit
                tuple((parse_bool, parse_and, alt((parse_expr_bu, parse_var_fcall)))),
                |(l, op, r)| (i, Expr::BinBOp(Box::new(l), op, Box::new(r))),
            ),
            map( // (expr) (*, /, %) unit
                tuple((preceded(tag("("), parse_expr_bool), preceded(tag(")"), parse_and), alt((parse_expr_bu, parse_var_fcall)))),
                |(l, op, r)| (i, Expr::BinBOp(Box::new(l), op, Box::new(r))),
            ),
            map( // Parses (expr) (bin op) expr
                tuple((preceded(tag("("), parse_expr_bool), preceded(tag(")"),parse_bop), parse_expr_bool)),
                |(l, op, r)| (i, Expr::BinBOp(Box::new(l), op, Box::new(r))),
            ),
            map( // Parses (expr) (bin op) expr
                tuple((preceded(tag("("), parse_expr_bool), preceded(tag(")"),parse_comp), parse_expr_bool)),
                |(l, op, r)| (i, Expr::Comp(Box::new(l), op, Box::new(r))),
            ),
            parse_expr_parentheses,
            map( // Parses !unit
                tuple((parse_not, alt((parse_expr_bu, parse_var_fcall)))),
                |(op, r)| (i, Expr::UBOp( op, Box::new(r))),
            ),
            parse_expr_comp,
            // Parses string to i32
            parse_bool,
        ))
    )(i)
}

fn parse_expr_comp_bool(i: Span) -> IResult<Span, SpanExpr> {
    preceded(multispace0,
        
        map(
                tuple((alt((parse_expr_parentheses,parse_bool)), parse_comp,
                alt((parse_expr_comp_bool,parse_expr_parentheses,parse_bool)))),
                |(l, op, r)| (i, Expr::Comp(Box::new(l), op, Box::new(r))),
        ),
    )(i)
}

fn parse_expr_comp(i: Span) -> IResult<Span, SpanExpr> {
    preceded(multispace0,
        alt((
            map(
                    tuple((alt((parse_expr_comp_bool,parse_expr_parentheses,parse_bool)), parse_comp, parse_expr)),
                    |(l, op, r)| (i, Expr::Comp(Box::new(l), op, Box::new(r))),
            ),
            parse_expr_comp_bool,
            map(
                    tuple((parse_expr_arith, parse_comp, parse_expr_comp)),
                    |(l, op, r)| (i, Expr::Comp(Box::new(l), op, Box::new(r))),
            ),
            map(
                    tuple((parse_expr_arith, parse_comp, parse_expr)),
                    |(l, op, r)| (i, Expr::Comp(Box::new(l), op, Box::new(r))),
            ),
        ))
    )(i)
}

// Parses arithmetic and boolean expressions
fn parse_expr(i: Span) -> IResult<Span, SpanExpr> {
    preceded(multispace0,
        alt((
            
            map(
                    tuple((parse_expr_bool, parse_bop, parse_expr_comp)),
                    |(l, op, r)| (i, Expr::BinBOp(Box::new(l), op, Box::new(r))),
            ),
            parse_expr_bool,
            map(
                    tuple((parse_expr_comp, parse_bop, parse_expr_bool)),
                    |(l, op, r)| (i, Expr::BinBOp(Box::new(l), op, Box::new(r))),
            ),
            parse_expr_comp,

            parse_expr_arith,
            
        ))
    )(i)
}

fn parse_expr_parentheses(i: Span) -> IResult<Span, SpanExpr>{
    preceded(multispace0, terminated(preceded(tag("("), parse_expr),preceded(multispace0,tag(")"))))(i)
}

#[derive(Copy, Clone ,Debug, PartialEq)]
pub enum Type{
    Int,
    Bool,
}

type SpanType<'a> = (Span<'a>, Type);




#[derive(Debug, PartialEq)]
pub enum Statement<'a> {
    Nil,
    VarDec(String, SpanType<'a>, Box::<SpanStatement<'a>>),
    VarAssign(String, Box::<SpanExpr<'a>>),
    //         Condition            If                          Else
    Condition(Box::<SpanExpr<'a>>, Box::<SpanStatement<'a>>, Box::<SpanStatement<'a>>),
    WhileLoop(Box::<SpanExpr<'a>>, Box::<SpanStatement<'a>>),
    FDef(String, SpanType<'a>, Box::<SpanStatement<'a>>, Box::<SpanStatement<'a>>),
    Expr(Box::<SpanExpr<'a>>),
    Node(Box::<SpanStatement<'a>>, Box::<SpanStatement<'a>>),
    Return(Box::<SpanExpr<'a>>),
}

type SpanStatement<'a> = (Span<'a>, Statement<'a>);

pub fn parse_outer_statement(i: Span) -> IResult<Span, SpanStatement> {
    preceded(multispace0, alt((
        // -------- fn def
            map(
                    tuple((parse_f_def, parse_outer_statement)),
                    |(l, r)| (i, Statement::Node(Box::new(l), Box::new(r)))
            ),
            parse_f_def,
    )))(i)
}

fn parse_statement(i: Span) -> IResult<Span, SpanStatement> {
    preceded(multispace0,
        alt((
            // -------- Var Declare
            map(
                tuple((terminated(parse_var_dec,preceded(multispace0,tag(";"))), parse_statement)),
                |(l, r)| (i, Statement::Node(Box::new(l), Box::new(r)))
            ),
            terminated(parse_var_dec,preceded(multispace0,tag(";"))),
            // -------- Var Assign
            map(
                    tuple((terminated(parse_var_assign,preceded(multispace0,tag(";"))), parse_statement)),
                    |(l, r)| (i, Statement::Node(Box::new(l), Box::new(r)))
            ),
            terminated(parse_var_assign,preceded(multispace0,tag(";"))),
            // -------- Condition
            map(
                    tuple((parse_condition, parse_statement)),
                    |(l, r)| (i, Statement::Node(Box::new(l), Box::new(r)))
            ),
            parse_condition,
            // -------- While loop
            map(
                    tuple((parse_while_loop, parse_statement)),
                    |(l, r)| (i, Statement::Node(Box::new(l), Box::new(r)))
            ),
            parse_while_loop,
            // -------- return
            parse_return,
            // -------- expr
            map(
                    tuple((terminated(parse_expr,preceded(multispace0,tag(";"))), parse_statement)),
                    |(l, r)| (i, Statement::Node(Box::new((Span::new(""),Statement::Expr(Box::new(l)))), Box::new(r)))
            ),
            map(
                    terminated(parse_expr,preceded(multispace0, tag(";"))),
                    |l| (i, Statement::Expr(Box::new(l)))
            ),
        ))
    )(i)
}

fn parse_return(i: Span) -> IResult<Span, SpanStatement>{
    preceded(multispace0,
        preceded(tag("return"),
            preceded(multispace1,
                alt((
                    map(
                        terminated(parse_expr,preceded(multispace0,tag(";"))),
                        |l| (i, Statement::Return(Box::new(l)))
                    ),
                    map(
                        parse_expr,
                        |l| (i, Statement::Return(Box::new(l)))
                    ),
                ))
            )
        )
    )(i)
}


fn parse_type(i: Span) -> IResult<Span, SpanType> {
    preceded(multispace0,
        alt((
            map(tag("i32"),|_|(i, Type::Int)),
            map(tag("bool"),|_|(i, Type::Bool)),
        ))
    )(i)
}

fn parse_var(i: Span) -> IResult<Span, String> {
        map(
            tuple((alpha1,alphanumeric0)),
            |(alpha_str,an_str):(Span,Span)| format!("{}{}",alpha_str.fragment,an_str.fragment)
        )(i)
}

fn parse_var_dec(i: Span) -> IResult<Span, SpanStatement> {
    preceded(multispace0,
        preceded(terminated(tag("let"),multispace1),

            alt((

                map(
                        tuple((terminated(parse_var,preceded(multispace0,tag(":"))), parse_type, preceded(preceded(multispace0,tag("=")), parse_var_dec_expr))),
                        |(v_name, v_type, val)| (i, Statement::VarDec(v_name, v_type, Box::new(val))),
                ),
                map(
                        tuple((terminated(parse_var,preceded(multispace0,tag(":"))), parse_type)),
                        |(v_name, v_type)| (i, Statement::VarDec(v_name, v_type, Box::new((Span::new(""),Statement::Nil)))),
                ),

            ))

        )
    )(i)
}

fn parse_var_dec_expr(i: Span) -> IResult<Span, SpanStatement> {
        map(
            parse_expr,
            |l| (i, Statement::Expr(Box::new(l)))
        )(i)
}

fn parse_var_assign(i: Span) -> IResult<Span, SpanStatement>{
    preceded(multispace0,
        map(
                    tuple((parse_var, preceded(preceded(multispace0,tag("=")), parse_expr))),
                    |(v_name, val)| (i, Statement::VarAssign(v_name, Box::new(val))),
            )
    )(i)
}

pub fn parse_brackets(i: Span) -> IResult<Span, SpanStatement>{
    preceded(multispace0, terminated(preceded(tag("{"), parse_statement),preceded(multispace0,tag("}"))))(i)
}

fn parse_condition(i: Span) -> IResult<Span, SpanStatement>{
    preceded(multispace0,
        preceded(tag("if"),
            alt((
                map(
                        tuple((alt((preceded(multispace1,parse_expr),parse_expr_parentheses)), parse_brackets, parse_else)),
                        |(cond, statement, else_statement)| (i, Statement::Condition(Box::new(cond), Box::new(statement), Box::new(else_statement))),
                ),
                map(
                        tuple((alt((preceded(multispace1,parse_expr),parse_expr_parentheses)), parse_brackets)),
                        |(cond, statement)| (i, Statement::Condition(Box::new(cond), Box::new(statement), Box::new((Span::new(""),Statement::Nil)))),
                )
            ))
        )
    )(i)
}

fn parse_else(i: Span) -> IResult<Span, SpanStatement>{
    preceded(multispace0,
        preceded(tag("else"),
            parse_brackets
        )
    )(i)
}

fn parse_while_loop(i: Span) -> IResult<Span, SpanStatement>{
    preceded(multispace0,
        preceded(tag("while"),
            map(
                    tuple((alt((preceded(multispace1,parse_expr),parse_expr_parentheses)), parse_brackets)),
                    |(cond, statement)| (i, Statement::WhileLoop(Box::new(cond), Box::new(statement))),
            )
        )
    )(i)
}

fn parse_f_def(i: Span) -> IResult<Span, SpanStatement>{
    preceded(multispace0,
        preceded(tag("fn"),
            map(
                    tuple((preceded(multispace1,parse_var), parse_parameters, preceded(preceded(multispace0,tag("->")),parse_type), parse_brackets)),
                    |(name, arg, r_type, statement)| (i, Statement::FDef(name, r_type, Box::new(arg), Box::new(statement))),
            )
        )
    )(i)
}

fn parse_parameters(i: Span) -> IResult<Span, SpanStatement>{
    preceded(multispace0,
        alt((
            terminated(parse_parameter,preceded(multispace0,tag(")"))), //5
            map(
                tuple((terminated(parse_parameter,preceded(multispace0,tag(","))), parse_parameters)), //4
                |(l,r)| (i, Statement::Node(Box::new(l), Box::new(r)))
            ),
            map(
                tuple((terminated(preceded(tag("("), parse_parameter),preceded(multispace0,tag(","))), parse_parameters)), //3
                |(l,r)| (i, Statement::Node(Box::new(l), Box::new(r)))
            ),
            terminated(preceded(tag("("), parse_parameter),preceded(multispace0,tag(")"))), //2
            map(//1
                tuple((tag("("), preceded(multispace0,tag(")")))),
                |(_,_)| (i, Statement::Nil)
            )
        ))
    )(i)
}

fn parse_parameter(i: Span) -> IResult<Span, SpanStatement> {
    preceded(multispace0,
        
        map(
            tuple((terminated(parse_var,tag(":")), parse_type)),
            |(v_name, v_type)| (i, Statement::VarDec(v_name, v_type, Box::new((Span::new(""),Statement::Nil)))),
        )

    )(i)
}

fn parse_f_call(i: Span) -> IResult<Span, SpanExpr>{
    preceded(multispace0,
        map(
                tuple((parse_var, tag("("), parse_arguments)),
                |(name, _, arg)| (i, Expr::FCall(name,Box::new(arg))),
        )
    )(i)
}

fn parse_arguments(i: Span) -> IResult<Span, SpanStatement>{
    preceded(multispace0,
        alt((
            terminated(parse_argument,preceded(multispace0,tag(")"))), //5
            map(
                tuple((terminated(parse_argument,preceded(multispace0,tag(","))), parse_arguments)), //4
                |(l,r)| (i, Statement::Node(Box::new(l), Box::new(r)))
            ),
            /*map(
                tuple((terminated(preceded(tag("("), parse_argument),preceded(multispace0,tag(","))), parse_arguments)), //3
                |(l,r)| (i, Statement::Node(Box::new(l), Box::new(r)))
            ),
            terminated(preceded(tag("("), parse_argument),preceded(multispace0,tag(")"))), //2*/
            map(//1
                tag(")"),
                |_| (i, Statement::Nil)
            )
        ))
    )(i)
}

fn parse_argument(i: Span) -> IResult<Span, SpanStatement> {
    preceded(multispace0,
        map(
            parse_expr,
            |l| (i, Statement::Expr(Box::new(l)))
        )
    )(i)
}

// Printing -----------


// dumps a Span into a String
fn dump_span(s: &Span) -> String {
    format!(
        "[line :{:?}, col:{:?}, {:?}]",
        s.line,
        s.get_column(),
        s.fragment
    )
}

fn dump_span_nofrag(s: &Span) -> String {
    format!(
        "line :{:?}, col:{:?}",
        s.line,
        s.get_column(),
    )
}

fn dump_statement(se: &SpanStatement) -> String {
    let (_, e) = se;
    match e {
        Statement::VarDec(st, t, v) => {
            format!("<{:?}: {} {} {}>", "VarDec:", st, dump_type(t), dump_statement(v))
        }
        Statement::VarAssign(st, v) => {
            format!("<{:?}: {} {}>", "VarrAssign:", st, dump_expr(v))
        }
        Statement::Condition(c, i, n) => {
            format!("<{:?}: {} {} {}>", "Condition:", dump_expr(c), dump_statement(i), dump_statement(n))
        }
        Statement::WhileLoop(c, state) => {
            format!("<{:?}: {} {}>", "WhileLoop:", dump_expr(c), dump_statement(state))
        }
        Statement::FDef(st, t, par, stat) => {
            format!("<{:?}: {} {} {} {}>", "FDef:", st, dump_type(t), dump_statement(par), dump_statement(stat))
        }
        Statement::Expr(expr) => {
            format!("<{:?}: {}>", "Expr:", dump_expr(expr))
        }
        Statement::Node(l, r) => {
            format!("<{:?}: {} {}>", "Node:", dump_statement(l), dump_statement(r))
        }
        Statement::Return(r) => {
            format!("<{:?}: {}>", "Return:", dump_expr(r))
        }
        Statement::Nil => {
            format!("<{:?}>", "Nil")
        }
    }
}


fn dump_type(st: &SpanType) -> String {
    let (s, e) = st;
    format!("[{:?}, {:?}]",dump_span_nofrag(s), e)
}



// dumps a SpanExpr into a String
fn dump_expr(se: &SpanExpr) -> String {
    let (s, e) = se;
    match e {
        Expr::Num(_) => dump_span(s),
        Expr::BinOp(l, (sop, _), r) => {
            format!("<{} {} {}>", dump_expr(l), dump_span(sop), dump_expr(r))
        }
        Expr::UOp( (sop, _), r) => {
            format!("<{} {} >",dump_span(sop), dump_expr(r))
        }
        Expr::Val(_) => dump_span(s),
        Expr::BinBOp(l, (sop, _), r) => {
            format!("<{} {} {}>", dump_expr(l), dump_span(sop), dump_expr(r))
        }
        Expr::UBOp( (sop, _), r) => {
            format!("<{} {} >",dump_span(sop), dump_expr(r))
        }
        Expr::Comp(l, (sop, _), r) => {
            format!("<{} {} {}>", dump_expr(l), dump_span(sop), dump_expr(r))
        }
        Expr::VarRef(_) => dump_span(s),
        Expr::FCall(_, ss) => {
            format!("<{} {}>", dump_span(s), dump_statement(ss))
        }
    }
}

// Type Checker ---------------------------------------------------------------

fn do_typechecking(fn_hmap: &HashMap<String, &Statement>) {
    for fnc in fn_hmap {
        typecheck_fdef(fn_hmap, fnc.1);
    }
}

fn typecheck_fdef(fn_hmap: &HashMap<String, &Statement>, fnc: &Statement) {

    match fnc {
        Statement::FDef(_,rt,param,bss) => {
            let mut env: Vec<HashMap<String, Type>> = Vec::new();
            env.push(HashMap::new());
            typecheck_param(fn_hmap, &mut env, param);
            if typecheck_statement(fn_hmap, &mut env, bss, rt.1) != (rt.1, true) {
                panic!("wrong return type")
            }
        }
        _ => panic!("not a function"),
    }
}

fn typecheck_param<'a>(fn_hmap: &HashMap<String, &Statement>, env: &mut Vec<HashMap<String, Type>>, s_param: &'a SpanStatement) {
    let i = env.len()-1;
    let (_, param) = s_param;
    match param {
        Statement::Nil => {
            return
        }
        Statement::Node(lp, rp) => {
            typecheck_param(fn_hmap, env, rp);
            typecheck_param(fn_hmap, env, lp);
        }
        Statement::VarDec(name, ts,_) => {
            let (_,t) = ts;
            env[i].insert(name.to_string(), *t);
        }
        _ => {
            panic!("Bad argument")
        }
    }
}

fn typecheck_statement(fn_hmap: &HashMap<String, &Statement>, env: &mut Vec<HashMap<String, Type>>, stmnt: &SpanStatement, ret_type: Type) -> (Type, bool) { // bool is in case of explicit return
    match &stmnt.1 {
        Statement::Condition(cond, then_ss, else_ss) => {
            if typecheck_expression(fn_hmap, env, &cond) != Type::Bool {
                panic!("does not eval to bool")
            }
            env.push(HashMap::new());
            let (rt,rb) = typecheck_statement(fn_hmap, env, &then_ss, ret_type);
            if rb == true {
                if rt != ret_type {
                    panic!("return does not match return type")
                } else {
                    return (rt,rb);
                }
            }
            env.pop();
            env.push(HashMap::new());
            let (rt,rb) = typecheck_statement(fn_hmap, env, &else_ss, ret_type);
            if rb == true {
                if rt != ret_type {
                    panic!("return does not match return type")
                } else {
                    return (rt,rb);
                }
            }
            env.pop();
            return (ret_type, false)
        }
        Statement::WhileLoop(cond, body_ss) => {
            if typecheck_expression(fn_hmap, env, &cond) != Type::Bool {
                panic!("does not eval to bool")
            }
            env.push(HashMap::new());
            let (rt,rb) = typecheck_statement(fn_hmap, env, &body_ss, ret_type);
            if rb == true {
                if rt != ret_type {
                    panic!("return does not match return type")
                } else {
                    return (rt,rb);
                }
            }
            env.pop();
            return (ret_type, false)
        }
        Statement::Expr(expr_ss) => {
            return (typecheck_expression(fn_hmap, env, expr_ss), false);
        }
        Statement::Nil => return (ret_type, false),
        Statement::Node(lss, rss) => {
            let (rt0,rb0) = typecheck_statement(fn_hmap, env, &lss, ret_type);
            let (rt1,rb1) = typecheck_statement(fn_hmap, env, &rss, ret_type);
            if (rb1 && (rt1 != ret_type)) || (rb0 && (rt0 != ret_type))  {
                panic!("returns wrong type")
            }
            return (ret_type, rb0 || rb1);
        }
        Statement::Return(se) => {
            let rt = typecheck_expression(fn_hmap, env, se);
            if rt != ret_type {
                panic!("returns wrong type")
            }
            return (ret_type, true);
        }
        Statement::VarDec(name,st, ss) => {
            let i = env.len()-1;
            env[i].insert(name.to_string(), st.1);
            let (rt, _) = typecheck_statement(fn_hmap, env, &ss, st.1);
            if rt != st.1 {
                panic!("cant assign wrong type")
            }
            return (rt,false);
        }
        Statement::VarAssign(name, e) => {
            let i = env.len()-1;
            let t = typecheck_var_ref(name,i, env);
            if t != typecheck_expression(fn_hmap, env, &e) {
                panic!("cant assign wrong type")
            }
            return (t, false);
        }
        _ => panic!("unexpected statement")
    }
}

fn typecheck_arg(fn_hmap: &HashMap<String, &Statement>, env: &mut Vec<HashMap<String, Type>>, param_ss: &SpanStatement,  args_ss: &SpanStatement) {
    match (&param_ss.1, &args_ss.1) {
        (Statement::Nil, Statement::Nil) => {
            return;
        }
        (Statement::Node(lp,rp), Statement::Node(la, ra)) => {
            typecheck_arg(fn_hmap, env, &lp, &la);
            typecheck_arg(fn_hmap, env, &rp, &ra);
        }
        (Statement::VarDec(_,st,_), Statement::Expr(se)) => {
            if st.1 != typecheck_expression(fn_hmap, env, &se) {
                panic!("wrong type!");
            }
        }
        (Statement::VarDec(_,st,_), _) => {
            if st.1 != typecheck_statement(fn_hmap, env, &args_ss, st.1).0 {
                panic!("wrong type!");
            }
        }
        _ => panic!("bad argument")
    }
}

fn typecheck_expression(fn_hmap: &HashMap<String, &Statement>, env: &mut Vec<HashMap<String, Type>>, expr: &SpanExpr) -> Type {
    match &expr.1 {
        Expr::Num(_) => {
            return Type::Int;
        }
        Expr::Val(_) => {
            return Type::Bool;
        }
        Expr::BinOp(l, _, r) => {
            match (typecheck_expression(fn_hmap, env, l), typecheck_expression(fn_hmap, env, r)) {
                (Type::Int, Type::Int) => {
                    return Type::Int;
                }
                _ => {
                    panic!("bad type")
                }
            };
        }
        Expr::UOp(_, r) => {
            match typecheck_expression(fn_hmap, env, r) {
                Type::Int => {
                    return Type::Int;
                }
                _ => {
                    panic!("bad type")
                }
            }
        }
        Expr::BinBOp(l, _, r) => {
            match (typecheck_expression(fn_hmap, env, l), typecheck_expression(fn_hmap, env, r)) {
                (Type::Bool, Type::Bool) => {
                    return Type::Bool;
                }
                _ => {
                    panic!("bad type")
                }
            };
        }
        Expr::UBOp(_, r) => {
            match typecheck_expression(fn_hmap, env, r) {
                Type::Bool => {
                    return Type::Bool;
                }
                _ => {
                    panic!("bad type")
                }
            }
        }
        Expr::Comp(l, _, r) => {
            match (typecheck_expression(fn_hmap, env, l), typecheck_expression(fn_hmap, env, r)) {
                (Type::Bool, Type::Bool) => {
                    return Type::Bool;
                }
                (Type::Int, Type::Int) => {
                    return Type::Bool;
                }
                _ => {
                    panic!("bad type")
                }
            };
        }
        Expr::VarRef(st) => {
            let i = env.len()-1;
            typecheck_var_ref(st,i, env)
            
        }
        Expr::FCall(name, arg_ss) => {
            match fn_hmap.get(name) {
                Some(fun) => {
                    match fun {
                        Statement::FDef(_,st,param_ss,_) => {
                            typecheck_arg(fn_hmap, env, param_ss, arg_ss);
                            return st.1
                        }
                        _ => panic!("not a function")
                    }
                }
                None => panic!("fn does not exist")
            }
        }
        
    }
}
fn typecheck_var_ref(name: &str, i:usize, env: &Vec<HashMap<String, Type>>) -> Type {
    match env[i].get(name) {
        Some(t) => {
            *t
        }
        None => {
            if i > 0 {
                typecheck_var_ref(name, i-1, env)
            }else{
                panic!("var not found")
            }
        }
    }
}

// Interpreter ----------------------------------------------------------------

#[derive(Copy, Clone, Debug, PartialEq)]
pub enum Val {
    Int(i32),
    Bool(bool),
    Nil,
}

fn eval_expr(fn_hmap: &HashMap<String, &Statement>, i: &SpanExpr, env: &Vec<HashMap<String, (Val, Type)>>) -> Val {
    let (_,e) = i;
    match e {
        Expr::Num(v) => {
            Val::Int(*v)
        }
        Expr::Val(v) => {
            Val::Bool(*v)
        }
        Expr::BinOp(l, (_, op), r) => {
            let (le, re) = match (eval_expr(fn_hmap, l, env),eval_expr(fn_hmap, r, env)) {
                (Val::Int(val), Val::Int(val2)) => {
                    (val, val2)
                }
                _ => {
                    panic!("typechecking fucked up")
                }
            };
            match op {
                Op::Add => { Val::Int(le + re) }
                Op::Mul => { Val::Int(le * re) }
                Op::Div => { Val::Int(le / re) }
                Op::Mod => { Val::Int(le % re) }
            }
        }
        Expr::UOp((_, uop), r) => {
            let re = match eval_expr(fn_hmap, r, env) {
                Val::Int(val) => {
                    val
                }
                _ => {
                    panic!("typechecking fucked up")
                }
            };
            match uop {
                UOp::Neg => {
                    Val::Int(-re)
                }
            }
        }
        Expr::BinBOp(l, (_, bop), r) => {
            let (le, re) = match (eval_expr(fn_hmap, l, env),eval_expr(fn_hmap, r, env)) {
                (Val::Bool(val), Val::Bool(val2)) => {
                    (val, val2)
                }
                _ => {
                    panic!("typechecking fucked up")
                }
            };
            match bop {
                BOp::And => {
                    Val::Bool(le && re)
                }
                BOp::Or => {
                    Val::Bool(le || re)
                }
            }
        }
        Expr::UBOp((_, ubop), r) => {
            let re = match eval_expr(fn_hmap, r, env) {
                Val::Bool(val) => {
                    val
                }
                _ => {
                    panic!("typechecking fucked up")
                }
            };
            match ubop {
                UBOp::Not => {
                    Val::Bool(!re)
                }
            }
        }
        Expr::Comp(l, (_, comp), r) => {
            let (le, re) = (eval_expr(fn_hmap, l, env),eval_expr(fn_hmap, r, env));
            match (le, re) {
                (Val::Bool(_), Val::Bool(_)) => {}
                (Val::Int(_), Val::Int(_)) => {}
                _ => {
                    panic!("typechecking fucked up")
                }
            };
            match comp {
                Comp::Equal => {
                    Val::Bool(le == re)
                }
                Comp::NotEqual => {
                    Val::Bool(le != re)
                }
            }
        }
        Expr::VarRef(st) => {
            let i = env.len()-1;
            var_ref(st,i, env)
            
        }
        Expr::FCall(name, s_arg) => {
            let mut vec: Vec<Val> = Vec::new();
            eval_arg(fn_hmap, s_arg, env, &mut vec);
            return interpret_fn(name, fn_hmap, &mut vec);
        }

    }
}

fn var_ref(name: &str, i:usize, env: &Vec<HashMap<String, (Val, Type)>>) -> Val {
    match env[i].get(name) {
        Some((v,_)) => {
            *v
        }
        None => {
            if i > 0 {
                var_ref(name, i-1, env)
            }else{
                panic!("var not found")
            }
        }
    }
}

fn build_fn_hash<'a>(i: &'a SpanStatement, mut hm: HashMap<String, &'a Statement<'a>>) -> HashMap<String, &'a Statement<'a>> {
    let (_, stmnt) = i;
    match stmnt {
        Statement::FDef(st, _, _, _) => {
            hm.insert((&st).to_owned().to_string(), stmnt);
            return hm;
        }
        Statement::Node(l,r) => {
            hm = build_fn_hash(l, hm);
            hm = build_fn_hash(r, hm);

            return hm;
        }
        _ => {
            panic!("not function or node")
        }
    }
}

fn interpret_fn(fn_name: &str, fn_hmap: &HashMap<String, &Statement>, args: &mut Vec<Val>) -> Val {
    let mut env: Vec<HashMap<String, (Val, Type)>>  = Vec::new();
    env.push(HashMap::new());
    let i = env.len()-1;
    let fnc = fn_hmap.get(fn_name).unwrap();
    let fn_par: &SpanStatement;
    let fn_body: &SpanStatement;
    let fn_type: &SpanType;
    match fnc {
        Statement::FDef(_, t, par, body) => {
            fn_par = par;
            fn_body = body;
            fn_type = t;
        }
        _ => {
            panic!("build_fn_hash f'd up");
        }
    }
    let (_, t) = fn_type;
    env[i].insert("return".to_owned(), (Val::Nil, *t));

    dec_param(fn_hmap, &mut env, fn_par, args);
    interpret_statement(fn_hmap, &mut env, fn_body);
    let (ret,_) = env[env.len()-1].get("return").unwrap();
    return *ret
}

fn dec_param<'a>(fn_hmap: &HashMap<String, &Statement>, env: &mut Vec<HashMap<String, (Val, Type)>>, s_param: &'a SpanStatement, args: &mut Vec<Val>){
    let i = env.len()-1;
    let (_, param) = s_param;
    match param {
        Statement::Nil => {
            return
        }
        Statement::Node(lp, rp) => {
            dec_param(fn_hmap, env, rp, args);
            dec_param(fn_hmap, env, lp, args);
        }
        Statement::VarDec(name, ts,_) => {
            let (_,t) = ts;
            let v = args.pop().unwrap();
            match v {
                Val::Bool(_) => {
                    if t == &Type::Bool{
                        env[i].insert(name.to_owned(), (v,*t));
                    }else {
                        panic!("type missmatch");
                    }
                }
                Val::Int(_) => {
                    if t == &Type::Int{
                        env[i].insert(name.to_owned(), (v,*t));
                    }else {
                        panic!("type missmatch");
                    }
                }
                Val::Nil => {
                    panic!("Nil argument not allowed");
                }
            }
        }
        _ => {
            panic!("Bad argument")
        }
    }
}

fn var_ass(name: &str, v: Val, env: &mut Vec<HashMap<String, (Val, Type)>>, i: usize){
    let temp = env[i].get(name);
    let temp_type: Type;
    match temp {
        Some((_, t)) => {
            temp_type = *t;
            env[i].insert(name.to_owned(), (v, temp_type));
        }
        None => {
            if i > 0 {
                var_ass(name, v, env, i-1);
            }else {
                panic!("Variable does not exist")
            }
        }
    }
}

fn interpret_statement<'a>(fn_hmap: &HashMap<String, &Statement>, env: &mut Vec<HashMap<String, (Val, Type)>>, body: &'a SpanStatement) -> bool{
    let i = env.len()-1;
    let (_, bdy) = body; // remove span
    match bdy {
        Statement::Expr(e) => {
            eval_expr(fn_hmap, e, env);
            return false;
        }
        Statement::VarDec(name,t,ss) => {
            let (_,at) = t;
            let temp_val: Val;
            match &ss.1 {
                Statement::Expr(e) => temp_val = eval_expr(fn_hmap, &e, env),
                Statement::Nil => temp_val = Val::Nil,
                _ => panic!("you can only assign an expression")
            };
            
            env[i].insert(name.to_owned(), (temp_val,*at));
            return false;
        }
        Statement::VarAssign(name, expr) => {
            let v = eval_expr(fn_hmap, expr, env);
            var_ass(name, v, env, i);
            return false;
        }
        Statement::Node(l,r) => {
            let lv = interpret_statement(fn_hmap, env, l);
            if lv {
                return true;
            }
            let rv = interpret_statement(fn_hmap, env, r);
            return rv
        }
        Statement::Return(e) => {
            let (_, t) = *env[0].get("return").unwrap();
            let temp_val: Val = eval_expr(fn_hmap, e, env);
            env[0].insert("return".to_owned(), (temp_val, t));
            return true;
        }
        Statement::Condition(cond_e, s_if, s_else) => {
            let c_val = eval_expr(fn_hmap, cond_e, env);
            let cond: bool;
            match c_val {
                Val::Bool(v) =>{
                    cond = v;
                }
                Val::Int(v) => {
                    cond = v > 0;
                }
                _ => {
                    panic!("Nil is not boolean")
                }
            }
            env.push(HashMap::new());
            let rv: bool;
            if cond {
                rv = interpret_statement(fn_hmap, env, s_if);
            }else{
                rv = interpret_statement(fn_hmap, env, s_else);
            }
            env.pop();
            return rv;
        }
        Statement::WhileLoop(cond, s_loop) => {
            env.push(HashMap::new());
            let rv = interpret_while(fn_hmap, env, s_loop, cond);
            env.pop();
            return rv;
        }
        Statement::Nil => {
            return false;
        }
        _ => panic!("Does not interpret FDef") 
        
    }
}

fn interpret_while(fn_hmap: &HashMap<String, &Statement>, env: &mut Vec<HashMap<String, (Val, Type)>>, s_loop: &SpanStatement, cond_se: &SpanExpr) -> bool {
    let c_val = eval_expr(fn_hmap, cond_se, env);
    let cond: bool;
    match c_val {
        Val::Bool(v) =>{
            cond = v;
        }
        Val::Int(v) => {
            cond = v > 0;
        }
        _ => {
            panic!("Nil is not boolean")
        }
    }
    if cond {
        let lv = interpret_statement(fn_hmap, env, s_loop);
        let rv = interpret_while(fn_hmap, env, s_loop, cond_se);
        return lv || rv;
    }else {
        return false;
    }
}
fn eval_arg(fn_hmap: &HashMap<String, &Statement>, s_arg: &SpanStatement, env: &Vec<HashMap<String, (Val, Type)>>, vec: &mut Vec<Val>) {
    match &s_arg.1 {
        Statement::Expr(e) => {
            vec.push(eval_expr(fn_hmap, e, env));
        }
        Statement::Node(l, r) => {
            eval_arg(fn_hmap, &l, env, vec);
            eval_arg(fn_hmap, &r, env, vec);
        }
        _ => {
            panic!("invalid arguments");
        }
    }
}

// compiler --------------------------------------------------------------------------

type ExprFunc = unsafe extern "C" fn() -> i32;

/// Compiler holds the LLVM state for the compilation
pub struct Compiler<'a> {
    pub context: &'a Context,
    pub builder: &'a Builder,
    // pub fpm: &'a PassManager<FunctionValue>,
    pub module: &'a Module,
    // pub function: &'a Func<'a>,
    variables: HashMap<String, PointerValue>,
    fn_value_opt: Option<FunctionValue>,
}

/// Compilation assumes the program to be semantically correct (well formed)
impl<'a> Compiler<'a> {
    /// Gets a defined function given its name.
    #[inline]
    fn get_function(&self, name: &str) -> Option<FunctionValue> {
        self.module.get_function(name)
    }

    /// Returns the `PointerValue` representing the variable `id`.
    #[inline]
    fn get_variable(&self, id: &str) -> &PointerValue {
        match self.variables.get(id) {
            Some(var) => var,
            None => panic!(
                "Could not find a matching variable, {} in {:?}",
                id, self.variables
            ),
        }
    }

    /// Returns the `FunctionValue` representing the function being compiled.
    #[inline]
    fn fn_value(&self) -> FunctionValue {
        self.fn_value_opt.unwrap()
    }

    /// For now, returns an IntValue
    /// Boolean is i1, single bit integers in LLVM
    /// However we might to choose to store them as i8 or i32
    fn compile_expr(&self, expr: &SpanExpr) -> IntValue {
        match &expr.1 {
            Expr::VarRef(s) => {
                let var = self.get_variable(&s);
                return self.builder.build_load(*var, &s).into_int_value();
            }
            Expr::FCall(name, s_arg) => {
                self.compile_fcall(name, s_arg)
            }
            Expr::Num(i) => self.context.i32_type().const_int(i.to_owned() as u64, false),

            Expr::BinOp(l, (_, op), r) => {
                let lv = self.compile_expr(&l);
                let rv = self.compile_expr(&r);
                match op {
                    Op::Add => self.builder.build_int_add(lv, rv, "sum"),
                    Op::Div => self.builder.build_int_signed_div(lv, rv, "div"),
                    Op::Mul => self.builder.build_int_mul(lv, rv, "prod"),
                    Op::Mod => self.builder.build_int_signed_rem(lv, rv, "mod"),
                }
            }
            Expr::BinBOp(l, (_, op), r) => {
                let lv = self.compile_expr(&l);
                let rv = self.compile_expr(&r);
                match op {
                    BOp::And => self.builder.build_and(lv, rv, "and"),
                    BOp::Or => self.builder.build_or(lv, rv, "or"),
                }
            }
            Expr::Comp(l, (_, op), r) => {
                let lv = self.compile_expr(&l);
                let rv = self.compile_expr(&r);
                match op {
                    Comp::Equal => self.builder.build_int_compare(inkwell::IntPredicate::EQ, lv, rv, "eq"),
                    Comp::NotEqual => self.builder.build_int_compare(inkwell::IntPredicate::NE, lv, rv, "eq"),
                }
            }
            Expr::UBOp((_, op), r) => {
                let rv = self.compile_expr(&r);
                match op {
                    UBOp::Not => self.builder.build_not(rv, "not"),
                }
            }
            Expr::UOp((_, op), r) => {
                let rv = self.compile_expr(&r);
                match op {
                    UOp::Neg => self.builder.build_int_neg(rv, "neg"),
                }
            }
            Expr::Val(b) => self.context.bool_type().const_int(b.to_owned() as u64, false),
            
            //_ => unimplemented!(),
        }
    }

    /// Creates a new stack allocation instruction in the entry block of the function.
    fn create_entry_block_alloca(&mut self, name: &str) -> PointerValue {
        let builder = self.context.create_builder();

        let entry = self.fn_value().get_first_basic_block().unwrap();

        match entry.get_first_instruction() {
            Some(first_instr) => builder.position_before(&first_instr),
            None => builder.position_at_end(&entry),
        }
        let alloca = builder.build_alloca(self.context.i32_type(), name);
        self.variables.insert(name.to_string(), alloca);
        alloca
    }

        

    /// Compiles a command into (InstructionValue, b:bool)
    /// `b` indicates that its a return value of the basic block
    fn compile_statement(&mut self, statement: &SpanStatement) -> (IntValue, bool) {
        match &statement.1 {
            Statement::Expr(se) => {
                (self.compile_expr(se), false)
            }
            Statement::VarAssign(name, e) => {
                let var = self.get_variable(&name);
                let rexp = self.compile_expr(&e);
                self.builder.build_store(*var, rexp);
                return (rexp, false);
            }
            Statement::VarDec(name, _, ss) => {
                let alloca = self.create_entry_block_alloca(&name);
                let temp_val: IntValue;
                match &ss.1 {
                    Statement::Expr(e) => temp_val = self.compile_expr(&e),
                    Statement::Nil => temp_val = self.context.i32_type().const_int(0,false),
                    _ => panic!("only nil and expr allowed for declaration")
                }
                self.builder.build_store(alloca, temp_val);
                return (temp_val, false);
            }
            Statement::Node(lss,rss) => {
                let ret = self.compile_statement(&lss);
                if ret.1 {
                    return ret;
                }
                self.compile_statement(&rss)
            }
            Statement::Return(e) => {
                let expr = self.compile_expr(&e);
                self.builder.build_return(Some(&expr));
                return (expr, true);
            }
            Statement::Nil => return (self.context.i32_type().const_int(0,false), false),
            Statement::Condition(cond_e, then_ss, else_ss) => {
                let parent = self.fn_value();
                let zero_const = self.context.i32_type().const_int(0,false);

                let cond = self.compile_expr(cond_e);
                let cond = self.builder.build_int_compare(IntPredicate::NE, cond, zero_const, "condition");

                // build branch
                let then_bb = self.context.append_basic_block(&parent, "then");
                let else_bb = self.context.append_basic_block(&parent, "else");
                let cont_bb = self.context.append_basic_block(&parent, "ifcont");

                self.builder.build_conditional_branch(cond, &then_bb, &else_bb);

                // build then block
                self.builder.position_at_end(&then_bb);
                let then_ret = self.compile_statement(then_ss);
                self.builder.build_unconditional_branch(&cont_bb);

                // build else block
                self.builder.position_at_end(&else_bb);
                let else_ret = self.compile_statement(else_ss);
                self.builder.build_unconditional_branch(&cont_bb);

                // emit merge block
                self.builder.position_at_end(&cont_bb);

                let phi = self.builder.build_phi(self.context.i32_type(), "iftmp");

                phi.add_incoming(&[
                    (&then_ret.0, &then_bb),
                    (&else_ret.0, &else_bb)
                ]);

                (phi.as_basic_value().into_int_value(), then_ret.1 || else_ret.1)
            }

            Statement::WhileLoop(cond_e, body_ss) => {
                let parent = self.fn_value();
                let zero_const = self.context.i32_type().const_int(0,false);

                let loop_head_bb = self.context.append_basic_block(&parent, "lhead");
                let loop_body_bb = self.context.append_basic_block(&parent, "lbody");
                let cont_bb = self.context.append_basic_block(&parent, "cont");

                self.builder.build_unconditional_branch(&loop_head_bb);
                self.builder.position_at_end(&loop_head_bb);
                let cond = self.compile_expr(cond_e);
                let cond = self.builder.build_int_compare(IntPredicate::NE, cond, zero_const, "cond");
                self.builder.build_conditional_branch(cond, &loop_body_bb, &cont_bb);

                self.builder.position_at_end(&loop_body_bb);
                let loop_ret = self.compile_statement(body_ss);
                self.builder.build_unconditional_branch(&loop_head_bb);

                self.builder.position_at_end(&cont_bb);

                return loop_ret;
            }
            
            
            _ => unimplemented!(),
        }
    }

    fn compile_program(&mut self, ss: &SpanStatement) {
        match &ss.1 {
            Statement::Node(lss, rss) => {
                self.compile_program(lss);
                self.compile_program(rss);
            }
            Statement::FDef(name, st, ss_par, ss_body) => {
                self.compile_fn(name, st, ss_par, ss_body);
            }
            _ => panic!("expected function or node")
        }
    }

    fn par_to_vec(&self, ss_par: &SpanStatement, vec: &mut Vec<(String, Type)>) {
        match &ss_par.1 {
            Statement::VarDec(name, s_type, _) => {
                vec.push((name.to_owned(), s_type.1));
            }
            Statement::Node(lpar, rpar) => {
                self.par_to_vec(lpar, vec);
                self.par_to_vec(rpar, vec);
            }
            Statement::Nil => return,
            _ => panic!("not a parameter")
        }
    }
    fn arg_to_vec(&self, ss_arg: &SpanStatement, vec: &mut Vec<BasicValueEnum>) {
        match &ss_arg.1 {
            Statement::Expr(e)  => {
                vec.push(self.compile_expr(&e).into());
            }
            Statement::Node(lss, rss) => {
                self.arg_to_vec(lss, vec);
                self.arg_to_vec(rss, vec);
            }
            _ => panic!("only expr fcall or nodes")
        }
    }
    fn compile_fcall(&self, name: &str, s_arg: &SpanStatement) -> IntValue {
        match self.get_function(name) {
            Some(fun) => {
                let mut arg_vec: Vec<BasicValueEnum> = Vec::new();
                self.arg_to_vec(s_arg, &mut arg_vec);

                match self.builder.build_call(fun, arg_vec.as_slice(), "tmp").try_as_basic_value().left() {
                    Some(val) => val.into_int_value(),
                    None => {
                        panic!("got no value");
                    }
                }
            }
            None => panic!("not found")
        }
    }
    fn compile_fn(&mut self, name: &str, _st: &SpanType, ss_par: &SpanStatement, ss_body: &SpanStatement) {
        let mut par_vec: Vec<(String, Type)> = Vec::new();
        self.par_to_vec(ss_par, &mut par_vec);
        let ret_type = self.context.i32_type();
        let par_types = std::iter::repeat(ret_type).take(par_vec.len()).map(|f| f.into()).collect::<Vec<BasicTypeEnum>>();
        let par_types = par_types.as_slice();
        let fn_type = self.context.i32_type().fn_type(par_types, false);
        let fn_val = self.module.add_function(name, fn_type, None);

        let entry = self.context.append_basic_block(&fn_val, "entry");
        self.builder.position_at_end(&entry);

        self.fn_value_opt = Some(fn_val);

        // build variables map
        self.variables.reserve(par_vec.len());

        for (i, arg) in fn_val.get_param_iter().enumerate() {
            let arg_name = par_vec[i].0.as_str();
            let alloca = self.create_entry_block_alloca(arg_name);

            self.builder.build_store(alloca, arg);

            self.variables.insert(par_vec[i].0.clone(), alloca);
        }

        self.compile_statement(ss_body);
    }

}

fn main() { // "fn f1() -> i32{let a : i32 = f2(5,3); a} fn f2(x: i32, y: i32) -> i32{return x*y}" "fn f1() -> i32{let a : i32 = 10;while a != 0 {a = a - 1;}a}
            // "fn f1() -> i32 {if true {return 1}}"
    let ss = parse_outer_statement(Span::new(
        /*"
            fn f1(a: i32) -> i32 {
                a = 2;
                return a;
            }
        "*/
        "
        fn f2(x: i32, y: i32) -> i32 {
            return x*y
        }
        fn f1() -> i32 {
            let a : i32 = f2(5,3);
            let b : i32 = 0;
            while b != 10 {
                b = b + 1;
            }
            if true && true {
                a = a + 3;
            } else {
                a = a + 5;
            }
            return a + b;
        }"
    )).unwrap().1;

    let hash_map = HashMap::new();
    let fn_hmap = build_fn_hash(&ss, hash_map);

    // Typechecker
    do_typechecking(&fn_hmap);

    // Compile into llvm ir
    compile(&ss);
    
    // Interpret program
    let mut args: Vec<Val> = Vec::new();
    println!("Result from interpreter: {:?}", interpret_fn("f1",&fn_hmap, &mut args));
    
    
}

fn compile(ss: &SpanStatement) {
    let context = Context::create();
    let module = context.create_module("f1");
    let builder = context.create_builder();
    let fpm = PassManager::create(&module);
    fpm.initialize();

    let u32_type = context.i32_type();
    let fn_type = u32_type.fn_type(&[], false);
    let function = module.add_function("f1", fn_type, None);
    let mut compiler = Compiler {
        context: &context,
        builder: &builder,
        module: &module,
        fn_value_opt: Some(function),
        variables: HashMap::new(),
        //&fpm,
    };

    compiler.compile_program(ss);


    module.print_to_stderr();
}
