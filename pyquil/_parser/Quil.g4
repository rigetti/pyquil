grammar Quil;

////////////////////
// PARSER
////////////////////

quil                : allInstr (NEWLINE+ allInstr)* NEWLINE* EOF ;

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
                    | nop
                    | include
                    | pragma
                    ;

// C. Static and Parametric Gates

gate                : name (LPAREN param (COMMA param)* RPAREN)? qubit+ ;

name                : IDENTIFIER ;
qubit               : UNSIGNED_INT ;

param               : dynamicParam | expression ;
dynamicParam        : LBRACKET qubit (MINUS qubit)? RBRACKET ;

// D. Gate Definitions

defGate             : DEFGATE name (LPAREN variable (COMMA variable)* RPAREN)? COLON NEWLINE matrix ;

variable            : PERCENTAGE IDENTIFIER ;

matrix              : (matrixRow NEWLINE) matrixRow ;
matrixRow           : TAB expression (COMMA expression)* ;

// E. Circuits

defCircuit          : DEFCIRCUIT name (LPAREN variable (COMMA variable)* RPAREN)? qubitVariable* COLON NEWLINE circuit ;

qubitVariable       : IDENTIFIER ;

circuitQubit        : qubit | qubitVariable ;
circuitGate         : name (LPAREN param (COMMA param)* RPAREN)? circuitQubit+ ;
circuitInstr        : circuitGate | instr ;
circuit             : (TAB circuitInstr NEWLINE)* TAB circuitInstr ;

// F. Measurement

measure             : MEASURE qubit addr? ;
addr                : LBRACKET classicalBit RBRACKET ;
classicalBit        : UNSIGNED_INT+ ;

// G. Program control

defLabel            : LABEL label ;
label               : AT IDENTIFIER ;
halt                : HALT ;
jump                : JUMP label ;
jumpWhen            : JUMPWHEN label addr ;
jumpUnless          : JUMPUNLESS label addr ;

// H. Zeroing the Quantum State

resetState          : RESET ; // NB: cannot be named "reset" due to conflict with Antlr implementation

// I. Classical/Quantum Synchronization

wait                : WAIT ;

// J. Classical Instructions

classicalUnary      : (TRUE | FALSE | NOT) addr ;
classicalBinary     : (AND | OR | MOVE | EXCHANGE) addr addr ;

// K. The No-Operation Instruction

nop                 : NOP ;

// L. File Inclusion

include             : INCLUDE STRING ;

// M. Pragma Support

pragma              : PRAGMA IDENTIFIER+ STRING? ;

// Expressions (in order of precedence)

expression          : LPAREN expression RPAREN                  #parenthesisExp
                    | <assoc=right> expression POWER expression #powerExp
                    | expression (TIMES | DIVIDE) expression    #mulDivExp
                    | expression (PLUS | MINUS) expression      #addSubExp
                    | function LPAREN expression RPAREN         #functionExp
                    | number                                    #numberExp
                    | variable                                  #variableExp
                    ;

function            : SIN | COS | SQRT | EXP | CIS ;

// Numbers

number              : realN | imaginaryN | I ;
imaginaryN          : realN I ;
realN               : floatN | intN ;

floatN              : (PLUS | MINUS)? UNSIGNED_FLOAT ;
intN                : (PLUS | MINUS)? UNSIGNED_INT ;

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

FALSE               : 'FALSE' ;
TRUE                : 'TRUE' ;
NOT                 : 'NOT' ;
AND                 : 'AND' ;
OR                  : 'OR' ;
MOVE                : 'MOVE' ;
EXCHANGE            : 'EXCHANGE' ;

PI                  : 'pi' ;
I                   : 'i' ;
SIN                 : 'sin' ;
COS                 : 'cos' ;
SQRT                : 'sqrt' ;
EXP                 : 'exp' ;
CIS                 : 'cis' ;

// Operators

PLUS                : '+' ;
MINUS               : '-' ; // Also serves as range in expressions like [8-71]
TIMES               : '*' ;
DIVIDE              : '/' ;
POWER               : '^' ;

// Identifiers

IDENTIFIER          : [A-Za-z] [A-Za-z0-9\-_]* ;

// Numbers

UNSIGNED_FLOAT      : UNSIGNED_INT PERIOD UNSIGNED_INT ;
UNSIGNED_INT        : [0-9]+ ;

// String

STRING              : QUOTE ~('\n' | '\r')* QUOTE;

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
NEWLINE             : ('\r'? '\n' | '\r')+ ;

// Skips

COMMENT             : '#' ~('\n' | '\r')* -> skip ;
SPACE               : ' ' -> skip ;
