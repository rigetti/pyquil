grammar Quil;

////////////////////
// PARSER
////////////////////

quil                : allInstr? ( NEWLINE+ allInstr )* NEWLINE* EOF ;

allInstr            : defGate
                    | defCircuit
                    | instr
                    ;

instr               : gate
                    | measure
                    | defLabel
                    | halt
                    | jump
                    | jumpWhen
                    | jumpUnless
                    | resetState
                    | wait
                    | classicalUnary
                    | classicalBinary
                    | classicalComparison
                    | load
                    | store
                    | nop
                    | include
                    | pragma
                    | memoryDescriptor
                    ;

// C. Static and Parametric Gates

gate                : name ( LPAREN param ( COMMA param )* RPAREN )? qubit+ ;

name                : IDENTIFIER ;
qubit               : INT ;

param               : expression ;

// D. Gate Definitions

defGate             : DEFGATE name ( LPAREN variable ( COMMA variable )* RPAREN )? COLON NEWLINE matrix ;

variable            : PERCENTAGE IDENTIFIER ;

matrix              : ( matrixRow NEWLINE )* matrixRow ;
matrixRow           : TAB expression ( COMMA expression )* ;

// E. Circuits

defCircuit          : DEFCIRCUIT name ( LPAREN variable ( COMMA variable )* RPAREN )? qubitVariable* COLON NEWLINE circuit ;

qubitVariable       : IDENTIFIER ;

circuitQubit        : qubit | qubitVariable ;
circuitGate         : name ( LPAREN param ( COMMA param )* RPAREN )? circuitQubit+ ;
circuitMeasure      : MEASURE circuitQubit addr? ;
circuitResetState   : RESET circuitQubit? ;
circuitInstr        : circuitGate | circuitMeasure | circuitResetState | instr ;
circuit             : ( TAB circuitInstr NEWLINE )* TAB circuitInstr ;

// F. Measurement

measure             : MEASURE qubit addr? ;
addr                : IDENTIFIER | ( IDENTIFIER? LBRACKET INT RBRACKET );

// G. Program control

defLabel            : LABEL label ;
label               : AT IDENTIFIER ;
halt                : HALT ;
jump                : JUMP label ;
jumpWhen            : JUMPWHEN label addr ;
jumpUnless          : JUMPUNLESS label addr ;

// H. Zeroing the Quantum State

resetState          : RESET qubit? ; // NB: cannot be named "reset" due to conflict with Antlr implementation

// I. Classical/Quantum Synchronization

wait                : WAIT ;

// J. Classical Instructions

memoryDescriptor    : DECLARE IDENTIFIER IDENTIFIER ( LBRACKET INT RBRACKET )? ( SHARING IDENTIFIER ( offsetDescriptor )* )? ;
offsetDescriptor    : OFFSET INT IDENTIFIER ;

classicalUnary      : ( NEG | NOT | TRUE | FALSE ) addr ;
classicalBinary     : logicalBinaryOp | arithmeticBinaryOp | move | exchange | convert ;
logicalBinaryOp     : ( AND | OR | IOR | XOR ) addr ( addr | INT ) ;
arithmeticBinaryOp  : ( ADD | SUB | MUL | DIV ) addr ( addr | number ) ;
move                : MOVE addr ( addr | number );
exchange            : EXCHANGE addr addr ;
convert             : CONVERT addr addr ;
load                : LOAD addr IDENTIFIER addr ;
store               : STORE IDENTIFIER addr ( addr | number );
classicalComparison : ( EQ | GT | GE | LT | LE ) addr addr ( addr | number );

// K. The No-Operation Instruction

nop                 : NOP ;

// L. File Inclusion

include             : INCLUDE STRING ;

// M. Pragma Support

pragma              : PRAGMA IDENTIFIER pragma_name* STRING? ;
pragma_name         : IDENTIFIER | INT ;

// Expressions (in order of precedence)

expression          : LPAREN expression RPAREN                  #parenthesisExp
                    | sign expression                           #signedExp
                    | <assoc=right> expression POWER expression #powerExp
                    | expression ( TIMES | DIVIDE ) expression  #mulDivExp
                    | expression ( PLUS | MINUS ) expression    #addSubExp
                    | function LPAREN expression RPAREN         #functionExp
                    | number                                    #numberExp
                    | variable                                  #variableExp
                    | addr                                      #addrExp
                    ;

function            : SIN | COS | SQRT | EXP | CIS ;
sign                : PLUS | MINUS ;

// Numbers
// We suffix -N onto these names so they don't conflict with already defined Python types

number              : realN | imaginaryN | I | PI ;
imaginaryN          : realN I ;
realN               : FLOAT | INT ;

////////////////////
// LEXER
////////////////////

// Keywords

DEFGATE             : 'DEFGATE' ;
DEFCIRCUIT          : 'DEFCIRCUIT' ;
MEASURE             : 'MEASURE' ;

LABEL               : 'LABEL' ;
HALT                : 'HALT' ;
JUMP                : 'JUMP' ;
JUMPWHEN            : 'JUMP-WHEN' ;
JUMPUNLESS          : 'JUMP-UNLESS' ;

RESET               : 'RESET' ;
WAIT                : 'WAIT' ;
NOP                 : 'NOP' ;
INCLUDE             : 'INCLUDE' ;
PRAGMA              : 'PRAGMA' ;

DECLARE             : 'DECLARE' ;
SHARING             : 'SHARING' ;
OFFSET              : 'OFFSET' ;

NEG                 : 'NEG' ;
NOT                 : 'NOT' ;
TRUE                : 'TRUE' ; // Deprecated
FALSE               : 'FALSE' ; // Deprecated

AND                 : 'AND' ;
IOR                 : 'IOR' ;
XOR                 : 'XOR' ;
OR                  : 'OR' ;   // Deprecated

ADD                 : 'ADD' ;
SUB                 : 'SUB' ;
MUL                 : 'MUL' ;
DIV                 : 'DIV' ;

MOVE                : 'MOVE' ;
EXCHANGE            : 'EXCHANGE' ;
CONVERT             : 'CONVERT' ;

EQ                  : 'EQ';
GT                  : 'GT';
GE                  : 'GE';
LT                  : 'LT';
LE                  : 'LE';

LOAD                : 'LOAD' ;
STORE               : 'STORE' ;

PI                  : 'pi' ;
I                   : 'i' ;

SIN                 : 'sin' ;
COS                 : 'cos' ;
SQRT                : 'sqrt' ;
EXP                 : 'exp' ;
CIS                 : 'cis' ;

// Operators

PLUS                : '+' ;
MINUS               : '-' ;
TIMES               : '*' ;
DIVIDE              : '/' ;
POWER               : '^' ;

// Identifiers

IDENTIFIER          : ( ( [A-Za-z_] ) | ( [A-Za-z_] [A-Za-z0-9\-_]* [A-Za-z0-9_] ) ) ;

// Numbers

INT                 : [0-9]+ ;
FLOAT               : [0-9]+ ( '.' [0-9]+ )? ( ( 'e'|'E' ) ( '+' | '-' )? [0-9]+ )? ;

// String

STRING              : '"' ~( '\n' | '\r' )* '"';

// Punctuation

PERIOD              : '.' ;
COMMA               : ',' ;
LPAREN              : '(' ;
RPAREN              : ')' ;
LBRACKET            : '[' ;
RBRACKET            : ']' ;
COLON               : ':' ;
PERCENTAGE          : '%' ;
AT                  : '@' ;
QUOTE               : '"' ;
UNDERSCORE          : '_' ;

// Whitespace

TAB                 : '    ' ;
NEWLINE             : ( '\r'? '\n' | '\r' )+ ;

// Skips

COMMENT             : '#' ~( '\n' | '\r' )* -> skip ;
SPACE               : ' ' -> skip ;

// Error

INVALID             : . ;