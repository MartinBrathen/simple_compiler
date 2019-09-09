extern crate nom;


use nom::{
    delimited,
    branch::alt,
    bytes::complete::{take_till,tag, take_while},
    character::complete::{digit1, multispace0, anychar},
    combinator::{map,map_res},
    sequence::{preceded, tuple, delimited, terminated},
    IResult,
};

use nom_locate::LocatedSpan;

type Span<'a> = LocatedSpan<&'a str>;

#[derive(Debug, PartialEq)]
pub enum Op {
    Add,
    Sub,
    Mul,
    Div,
    Mod,
}

type SpanOp<'a> = (Span<'a>, Op);

fn parse_add_sub(i: Span) -> IResult<Span, SpanOp> {
    alt((
        map(tag("+"), |s| (s, Op::Add)),
        map(tag("-"), |s| (s, Op::Sub)),
        
    ))(i)
}

fn parse_mdm(i: Span) -> IResult<Span, SpanOp> {
    alt((
        map(tag("*"), |s| (s, Op::Mul)),
        map(tag("/"), |s| (s, Op::Div)),
        map(tag("%"), |s| (s, Op::Mod)),
    ))(i)
}

fn parse_op(i: Span) -> IResult<Span, SpanOp> {
    alt((
        map(tag("*"), |s| (s, Op::Mul)),
        map(tag("/"), |s| (s, Op::Div)),
        map(tag("%"), |s| (s, Op::Mod)),
        map(tag("+"), |s| (s, Op::Add)),
        map(tag("-"), |s| (s, Op::Sub)),
    ))(i)
}

#[derive(Debug, PartialEq)]
pub enum Expr<'a> {
    Num(i32),
    BinOp(Box<SpanExpr<'a>>, SpanOp<'a>, Box<SpanExpr<'a>>),
}

type SpanExpr<'a> = (Span<'a>, Expr<'a>);

pub fn parse_i32(i: Span) -> IResult<Span, SpanExpr> {
    map(digit1, |digit_str: Span| {
        (
            digit_str,
            Expr::Num(digit_str.fragment.parse::<i32>().unwrap()),
        )
    })(i)
}

fn parse_expr(i: Span) -> IResult<Span, SpanExpr> {
    alt((
        map( // Parses num (+, -) expr
            tuple((parse_i32, preceded(multispace0, parse_add_sub), parse_expr_ms)),
            |(l, op, r)| (i, Expr::BinOp(Box::new(l), op, Box::new(r))),
        ),
        map( // Parses num (bin_op) num (bin op) expr
            tuple((parse_expr_mdm, preceded(multispace0, parse_op), parse_expr_ms)),
            |(l, op, r)| (i, Expr::BinOp(Box::new(l), op, Box::new(r))),
        ),
            // Parses num (bin_op) num 
        parse_expr_mdm,
//-----------------------------------------------------------------------------------------------------------------------------------------



        /* map( // Parses (expr (bin op) expr) (bin op) expr
            tuple((preceded(tag("("), parse_expr_ms), preceded(tag(")"),preceded(multispace0, parse_op)), parse_expr_mdm)),
            |(l, op, r)| (i, Expr::BinOp(Box::new(l), op, Box::new(r))),
            //tuple((remove_parenthesis, preceded(multispace0, parse_op), parse_expr_ms)),
            //|(l, op, r)| (i, Expr::BinOp(Box::new(parse_expr_ms(l).unwrap().1), op, Box::new(r))),
        ), */
        map( // Parses (expr (bin op) expr) (bin op) expr
            tuple((preceded(tag("("), parse_expr_ms), preceded(tag(")"),preceded(multispace0, parse_op)), parse_expr_ms)),
            |(l, op, r)| (i, Expr::BinOp(Box::new(l), op, Box::new(r))),
            //tuple((remove_parenthesis, preceded(multispace0, parse_op), parse_expr_ms)),
            //|(l, op, r)| (i, Expr::BinOp(Box::new(parse_expr_ms(l).unwrap().1), op, Box::new(r))),
        ),
        map( // Parses (expr (bin op) expr) (bin op) expr
            tuple((preceded(tag("("), parse_expr_ms), preceded(tag(")"),preceded(multispace0, parse_op)), preceded(multispace0, parse_i32))),
            |(l, op, r)| (i, Expr::BinOp(Box::new(l), op, Box::new(r))),
            //tuple((remove_parenthesis, preceded(multispace0, parse_op), parse_expr_ms)),
            //|(l, op, r)| (i, Expr::BinOp(Box::new(parse_expr_ms(l).unwrap().1), op, Box::new(r))),
        ),
        
        terminated(preceded(tag("("), parse_expr_ms),tag(")")),
        //map( // Parses (expr (bin op) expr)
            //remove_parenthesis,
            //|l| parse_expr_ms(l).unwrap().1,
        //),
            // Parses a single num
        parse_i32,
    ))(i)
}

fn parse_expr_mdm(i: Span) -> IResult<Span, SpanExpr>{
    alt((
        map(
            tuple((parse_i32, preceded(multispace0, parse_mdm), preceded(multispace0, parse_expr_mdm))),
            |(l, op, r)| (i, Expr::BinOp(Box::new(l), op, Box::new(r))),
        ),
        map(
            tuple((preceded(tag("("), parse_expr_ms), preceded(tag(")"),preceded(multispace0, parse_mdm)), preceded(multispace0, parse_expr_mdm))),
            |(l, op, r)| (i, Expr::BinOp(Box::new(l), op, Box::new(r))),
        ),
        terminated(preceded(tag("("), parse_expr_ms),tag(")")),
        parse_i32,
    ))(i)
}


fn parse_expr_ms(i: Span) -> IResult<Span, SpanExpr> {
    preceded(multispace0, parse_expr)(i)
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

// dumps a SpanExpr into a String
fn dump_expr(se: &SpanExpr) -> String {
    let (s, e) = se;
    match e {
        Expr::Num(_) => dump_span(s),
        Expr::BinOp(l, (sop, _), r) => {
            format!("<{} {} {}>", dump_expr(l), dump_span(sop), dump_expr(r))
        }
    }
}

fn main() {
    let (_, (s, e)) = parse_expr_ms(Span::new("(1*(1+2)+1)*((2+1)*4)")).unwrap();
    println!(
        "span for the whole,expression: {:?}, \nline: {:?}, \ncolumn: {:?}",
        s,
        s.line,
        s.get_column()
    );

    println!("raw e: {:?}", &e);
    println!("pretty e: {}", dump_expr(&(s, e)));
}

// In this example, we have a `parse_expr_ms` is the "top" level parser.
// It consumes white spaces, allowing the location information to reflect the exact
// positions in the input file.
//
// The dump_expr will create a pretty printing of the expression with spans for
// each terminal. This will be useful for later for precise type error reporting.
//
// The extra field is not used, it can be used for metadata, such as filename.
