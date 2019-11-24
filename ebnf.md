# EBNF

program\
: { function_def}\
;

function_def\
: name '(' [name ':' type] { ',' name ':' type } ')' '{' { statement } '}'\
;

statement\
: (var_dec\
| var_assign\
| condition\
| loop ) ';'\
| return [';']\
| expr ';'\
;

condition\
: 'if' white_space expr '{' {statement} '}' [ 'else' '{' statement '}' ]\
;

loop\
: 'while' white_space expr '{' {statement} '}'\
;

return\
: 'return' white_space expr\
| expr\
;

var_assign\
: name '=' expr\
;

var_dec\
: 'let', white_space name ':' type ['=', expr,';']\
;

expr \
: expr_bool | expr_int\
;

expr_int \
: int\
| name\
| expr_int '+' expr_int\
| expr_int ('*' | '/' | '%') expr_int\
| '(' expr_int ')'\
;

expr_bool \
: bool\
| name\
| expr_bool '||' expr_bool\
| expr_bool '&&' expr_bool\
| expr_bool ('!=' | '==') expr_bool\
| expr_int ('!=' | '==') expr_int\
| '(' expr_bool ')'\
;

name\
: char, {digit | char}\
;

char\
: "A" | "B" | "C" | "D" | "E" | "F" | "G"
| "H" | "I" | "J" | "K" | "L" | "M" | "N"
| "O" | "P" | "Q" | "R" | "S" | "T" | "U"
| "V" | "W" | "X" | "Y" | "Z" | "a" | "b"
| "c" | "d" | "e" | "f" | "g" | "h" | "i"
| "j" | "k" | "l" | "m" | "n" | "o" | "p"
| "q" | "r" | "s" | "t" | "u" | "v" | "w"
| "x" | "y" | "z"\
;

digit\
:
"0" | "1" | "2" | "3" | "4" | "5" | "6" | "7" | "8" | "9"\
;

int\
: ('+'|'-') digit {digit}\
;


bool\
: 'true' | 'false'\
;

white_space\
: ' \'\
| '\n'\
| '\t'\
;