extern crate nom;

use nom::{
    branch::alt,
    bytes::complete::{tag,},
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

fn parse_var_ref(i: Span) -> IResult<Span, SpanExpr> {
        map(
            tuple((alpha1,alphanumeric0)),
            |(alpha_str,an_str):(Span,Span)| (i,Expr::VarRef(format!("{}{}",alpha_str.fragment,an_str.fragment)))
        )(i)
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
            parse_var_ref,
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
                tuple((alt((parse_expr_bu, parse_var_ref)), parse_bop, parse_expr_bool)),
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
                tuple((parse_bool, parse_and, alt((parse_expr_bu, parse_var_ref)))),
                |(l, op, r)| (i, Expr::BinBOp(Box::new(l), op, Box::new(r))),
            ),
            map( // (expr) (*, /, %) unit
                tuple((preceded(tag("("), parse_expr_bool), preceded(tag(")"), parse_and), alt((parse_expr_bu, parse_var_ref)))),
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
                tuple((parse_not, alt((parse_expr_bu, parse_var_ref)))),
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

fn parse_statement_parentheses(i: Span) -> IResult<Span, SpanStatement>{
    preceded(multispace0, terminated(preceded(tag("("), parse_statement_returning),preceded(multispace0,tag(")"))))(i)
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
    VarAssign(String, Box::<SpanStatement<'a>>),
    //         Condition            If                          Else
    Condition(Box::<SpanStatement<'a>>, Box::<SpanStatement<'a>>, Box::<SpanStatement<'a>>),
    WhileLoop(Box::<SpanStatement<'a>>, Box::<SpanStatement<'a>>),
    FDef(String, SpanType<'a>, Box::<SpanStatement<'a>>, Box::<SpanStatement<'a>>),
    FCall(String, Box::<SpanStatement<'a>>),
    Expr(Box::<SpanExpr<'a>>),
    Node(Box::<SpanStatement<'a>>, Box::<SpanStatement<'a>>),
    Return(Box::<SpanStatement<'a>>),
}

type SpanStatement<'a> = (Span<'a>, Statement<'a>);

fn parse_outer_statement(i: Span) -> IResult<Span, SpanStatement> {
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
            // -------- fn call
            map(
                    tuple((terminated(parse_f_call,preceded(multispace0,tag(";"))), parse_statement)),
                    |(l, r)| (i, Statement::Node(Box::new(l), Box::new(r)))
            ),
            // -------- expr
            map(
                    tuple((terminated(parse_expr,preceded(multispace0,tag(";"))), parse_statement)),
                    |(l, r)| (i, Statement::Node(Box::new((Span::new(""),Statement::Expr(Box::new(l)))), Box::new(r)))
            ),
            map(
                    terminated(parse_expr,preceded(multispace0, tag(";"))),
                    |l| (i, Statement::Return(Box::new((Span::new(""),Statement::Expr(Box::new(l))))))
            ),
            // -------- Statements that return a value
            parse_statement_returning,
        ))
    )(i)
}

fn parse_statement_returning(i: Span) -> IResult<Span, SpanStatement> {
    preceded(multispace0,
        alt((
            // -------- return
            map(
                    terminated(parse_return,preceded(multispace0,tag(";"))),
                    |l| (i, Statement::Return(Box::new(l)))
            ),
            parse_return,
            // -------- fn call
            map(
                    parse_f_call,
                    |r| (i, Statement::Return(Box::new(r)))
            ),
            // -------- expr
            map(
                    parse_expr,
                    |l| (i, Statement::Return(Box::new((Span::new(""),Statement::Expr(Box::new(l))))))
            ),
        ))
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
                        tuple((terminated(parse_var,preceded(multispace0,tag(":"))), parse_type, preceded(preceded(multispace0,tag("=")), parse_statement_returning))),
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

fn parse_var_assign(i: Span) -> IResult<Span, SpanStatement>{
    preceded(multispace0,
        map(
                    tuple((parse_var, preceded(preceded(multispace0,tag("=")), parse_statement_returning))),
                    |(v_name, val)| (i, Statement::VarAssign(v_name, Box::new(val))),
            )
    )(i)
}

fn parse_brackets(i: Span) -> IResult<Span, SpanStatement>{
    preceded(multispace0, terminated(preceded(tag("{"), parse_statement),preceded(multispace0,tag("}"))))(i)
}

fn parse_condition(i: Span) -> IResult<Span, SpanStatement>{
    preceded(multispace0,
        preceded(tag("if"),
            alt((
                map(
                        tuple((alt((preceded(multispace1,parse_statement),parse_statement_parentheses)), parse_brackets, parse_else)),
                        |(cond, statement, else_statement)| (i, Statement::Condition(Box::new(cond), Box::new(statement), Box::new(else_statement))),
                ),
                map(
                        tuple((alt((preceded(multispace1,parse_statement),parse_statement_parentheses)), parse_brackets)),
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
                    tuple((alt((preceded(multispace1,parse_statement),parse_statement_parentheses)), parse_brackets)),
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

fn parse_f_call(i: Span) -> IResult<Span, SpanStatement>{
    preceded(multispace0,
        map(
                tuple((parse_var, parse_arguments)),
                |(name, arg)| (i, Statement::FCall(name,Box::new(arg))),
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
            map(
                tuple((terminated(preceded(tag("("), parse_argument),preceded(multispace0,tag(","))), parse_arguments)), //3
                |(l,r)| (i, Statement::Node(Box::new(l), Box::new(r)))
            ),
            terminated(preceded(tag("("), parse_argument),preceded(multispace0,tag(")"))), //2
            map(//1
                tuple((tag("("), preceded(multispace0,tag(")")))),
                |(_,_)| (i, Statement::Nil)
            )
        ))
    )(i)
}

fn parse_argument(i: Span) -> IResult<Span, SpanStatement> {
    preceded(multispace0,
        alt((
            parse_f_call,
            map(
                parse_expr,
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
                    terminated(parse_statement_returning,preceded(multispace0, tag(";"))),
                    parse_statement_returning,
                ))
            )
        )
    )(i)
}


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
            format!("<{:?}: {} {}>", "VarrAssign:", st, dump_statement(v))
        }
        Statement::Condition(c, i, n) => {
            format!("<{:?}: {} {} {}>", "Condition:", dump_statement(c), dump_statement(i), dump_statement(n))
        }
        Statement::WhileLoop(c, state) => {
            format!("<{:?}: {} {}>", "WhileLoop:", dump_statement(c), dump_statement(state))
        }
        Statement::FDef(st, t, par, stat) => {
            format!("<{:?}: {} {} {} {}>", "FDef:", st, dump_type(t), dump_statement(par), dump_statement(stat))
        }
        Statement::FCall(st, arg) => {
            format!("<{:?}: {} {}>", "FCall:", st, dump_statement(arg))
        }
        Statement::Expr(expr) => {
            format!("<{:?}: {}>", "Expr:", dump_expr(expr))
        }
        Statement::Node(l, r) => {
            format!("<{:?}: {} {}>", "Node:", dump_statement(l), dump_statement(r))
        }
        Statement::Return(r) => {
            format!("<{:?}: {}>", "Return:", dump_statement(r))
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

    }
}

#[derive(Copy, Clone, Debug, PartialEq)]
pub enum Val {
    Int(i32),
    Bool(bool),
    Nil,
}


fn build_fn_hash_test<'a>(stmnt: Statement<'a>, hm: &mut HashMap<String, (Type, Statement<'a>, Statement<'a>)>){
    match stmnt {
        Statement::FDef(name, s_type, s_par, s_st) => {
            hm.insert(name, (s_type.1, s_par.1, s_st.1));
        }
        Statement::Node(l,r) => {
            build_fn_hash_test(l.1, hm);
            build_fn_hash_test(r.1, hm);
        }
        _ => {
            panic!("not function or node")
        }
    }

}


fn eval_expr(i: &SpanExpr, env: &Vec<HashMap<String, (Val, Type)>>) -> Val {
    let (_,e) = i;
    match e {
        Expr::Num(v) => {
            Val::Int(*v)
        }
        Expr::Val(v) => {
            Val::Bool(*v)
        }
        Expr::BinOp(l, (_, op), r) => {
            let (le, re) = match (eval_expr(l, env),eval_expr(r, env)) {
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
            let re = match eval_expr(r, env) {
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
            let (le, re) = match (eval_expr(l, env),eval_expr(r, env)) {
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
            let re = match eval_expr(r, env) {
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
            let (le, re) = (eval_expr(l, env),eval_expr(r, env));
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

}fn interpret_fn(fn_name: &str, fn_hmap: &HashMap<String, &Statement>, args: &mut Vec<Val>) -> Val {
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
    println!("do i get here?");
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

fn interpret_statement<'a>(fn_hmap: &HashMap<String, &Statement>, env: &mut Vec<HashMap<String, (Val, Type)>>, body: &'a SpanStatement){
    let i = env.len()-1;
    let (_, bdy) = body; // remove span
    match bdy {
        Statement::Expr(e) => {
            eval_expr(e, env);
        }
        Statement::VarDec(name,t,s) => {
            let (_,at) = t;
            let temp_val: Val = eval_fcall_or_expr(fn_hmap, env, s);
            env[i].insert(name.to_owned(), (temp_val,*at));
        }
        Statement::VarAssign(name,s) => {
            let v = eval_fcall_or_expr(fn_hmap, env, s);
            var_ass(name, v, env, i);
        }
        Statement::Node(l,r) => {
            interpret_statement(fn_hmap, env, l);
            interpret_statement(fn_hmap, env, r);
        }
        Statement::FCall(name, s_arg) => {
            let mut vec: Vec<Val> = Vec::new();
            eval_arg(fn_hmap, s_arg, env, &mut vec);
            interpret_fn(name, fn_hmap, &mut vec);
        }
        Statement::Return(s) => {
            println!("{:?}",&s);
            let (_, t) = *env[0].get("return").unwrap();
            let temp_val: Val = eval_fcall_or_expr(fn_hmap, env, s);
            env[0].insert("return".to_owned(), (temp_val, t));
            return;
        }
        Statement::Condition(s_cond, s_if, s_else) => {
            let c_val = eval_fcall_or_expr(fn_hmap, env, s_cond);
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
            if cond {
                interpret_statement(fn_hmap, env, s_if);
            }else{
                interpret_statement(fn_hmap, env, s_else);
            }
            env.pop();
        }
        Statement::WhileLoop(s_cond, s_loop) => {
            env.push(HashMap::new());
            interpret_while(fn_hmap, env, s_loop, s_cond);
            env.pop();
        }
        Statement::Nil => {
            return;
        }
        _ => { // does not support loops and ifs yeet
            panic!("Does not interpret FDef and Nil") 
        }
    }
}


fn interpret_while(fn_hmap: &HashMap<String, &Statement>, env: &mut Vec<HashMap<String, (Val, Type)>>, s_loop: &SpanStatement, s_cond: &SpanStatement) {
    let c_val = eval_fcall_or_expr(fn_hmap, env, s_cond);
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
        interpret_statement(fn_hmap, env, s_loop);
        interpret_while(fn_hmap, env, s_loop, s_cond);
    }else {
        return
    }
}

fn eval_arg(fn_hmap: &HashMap<String, &Statement>, s_arg: &SpanStatement, env: &Vec<HashMap<String, (Val, Type)>>, vec: &mut Vec<Val>) {
    match &s_arg.1 {
        Statement::Expr(_) | Statement::FCall(_,_) => {
            vec.push(eval_fcall_or_expr(fn_hmap, env, s_arg));
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


fn eval_fcall_or_expr(fn_hmap: &HashMap<String, &Statement>, env: &Vec<HashMap<String, (Val, Type)>>, body: &SpanStatement) -> Val {
    let (_, s) = body;
    match s {
        Statement::Expr(e) => {
            return eval_expr(e, env);
        }
        Statement::FCall(st, arg) => {
            let mut vec: Vec<Val> = Vec::new();
            eval_arg(fn_hmap, arg, env, &mut vec);
            return interpret_fn(&st, fn_hmap, &mut vec)
        }
        Statement::Return(ss) => {
            eval_fcall_or_expr(fn_hmap, env, ss)
        }
        _ => {
            panic!("Can only take expr, fcall or return as argument")
        }
    }
}


fn main() { // "fn f1() -> i32{let a : i32 = f2(5,3); a} fn f2(x: i32, y: i32) -> i32{return x*y}" "fn f1() -> i32{let a : i32 = 10;while a != 0 {a = a - 1;}a}
            // "fn f1() -> i32 {if true {return 1}}"
    let (_, (s, e)) = parse_outer_statement(Span::new("fn f1() -> i32{let arg : i32 = 5;let a : i32 = f2(5,arg); a} fn f2(x: i32, y: i32) -> i32{return x*2 + y}")).unwrap();
    //println!("{:?} ", e)
    let hash_map = HashMap::new();
    let mut args: Vec<Val> = Vec::new();
    println!("{:?}", interpret_fn("f1",&build_fn_hash(&(s,e), hash_map), &mut args));
}

// In this example, we have a `parse_expr_ms` is the "top" level parser.
// It consumes white spaces, allowing the location information to reflect the exact
// positions in the input file.
//
// The dump_expr will create a pretty printing of the expression with spans for
// each terminal. This will be useful for later for precise type error reporting.
//
// The extra field is not used, it can be used for metadata, such as filename.

// TODO: Fix return parsing(low prio)
// Bug: (x+1)*2 gave error