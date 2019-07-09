grammar Quil;

////////////////////
// PARSER
////////////////////

quil                : allInstr? ( NEWLINE+ allInstr )* NEWLINE* EOF ;

allInstr            : defGate
                    | defGateAsPauli
                    | defCircuit
                    | defFrame
                    | defWaveform
                    | defCalibration
                    | defMeasCalibration
                    | instr
                    ;

instr               : fence
                    | delay
                    | gate
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
                    | pulse
                    | setFrequency
                    | setPhase
                    | shiftPhase
                    | swapPhase
                    | setScale
                    | capture
                    | rawCapture
                    | memoryDescriptor // this is a little unusual, but it's in steven's example
                    ;

// C. Static and Parametric Gates

gate                : modifier* name ( LPAREN param ( COMMA param )* RPAREN )? qubit+ ;

name                : IDENTIFIER ;
qubit               : INT ;

param               : expression ;

modifier            : CONTROLLED
                    | DAGGER
                    | FORKED ;

// D. Gate Definitions

defGate             : DEFGATE name (( LPAREN variable ( COMMA variable )* RPAREN ) | ( AS gatetype ))? COLON NEWLINE matrix ;
defGateAsPauli      : DEFGATE name ( LPAREN variable ( COMMA variable )* RPAREN )? qubitVariable+ AS PAULISUM COLON NEWLINE pauliTerms ;

variable            : PERCENTAGE IDENTIFIER ;
gatetype            : MATRIX
                    | PERMUTATION ;

matrix              : ( matrixRow NEWLINE )* matrixRow ;
matrixRow           : TAB expression ( COMMA expression )* ;

pauliTerms          : ( pauliTerm NEWLINE )* pauliTerm;
pauliTerm           : TAB IDENTIFIER LPAREN expression RPAREN qubitVariable+ ;

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

pragma              : PRAGMA ( IDENTIFIER | keyword ) pragma_name* STRING? ;
pragma_name         : IDENTIFIER | keyword | INT ;

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

number              : MINUS? ( realN | imaginaryN | I | PI ) ;
imaginaryN          : realN I ;
realN               : FLOAT | INT ;

// Analog control

defFrame            : DEFFRAME frame ( COLON frameSpec+ )? ;
frameSpec           : NEWLINE TAB frameAttr COLON ( expression | STRING ) ;
frameAttr           : SAMPLERATE | INITIALFREQUENCY | DIRECTION ;

defWaveform         : DEFWAVEFORM name ( LPAREN param (COMMA param)* RPAREN )? realN COLON NEWLINE matrix ;
defCalibration      : DEFCAL name (LPAREN param ( COMMA param )* RPAREN)? formalQubit+ COLON ( NEWLINE TAB instr )* ;
defMeasCalibration  : DEFCAL MEASURE formalQubit ( name )? COLON ( NEWLINE TAB instr )* ;

pulse               : NONBLOCKING? PULSE frame waveform ;
capture             : NONBLOCKING? CAPTURE frame waveform addr ; // TODO: augment this to parse affine kernels
rawCapture          : NONBLOCKING? RAWCAPTURE frame expression addr ;

setFrequency        : SETFREQUENCY frame expression ;
setPhase            : SETPHASE frame expression ;
shiftPhase          : SHIFTPHASE frame expression ;
swapPhase           : SWAPPHASE frame frame ;
setScale            : SETSCALE frame expression ;

delay               : DELAY formalQubit+ STRING* expression ;
fence               : FENCE formalQubit+ ;

formalQubit         : qubit | qubitVariable ;
namedParam          : IDENTIFIER COLON expression ;
waveform            : name (LPAREN namedParam ( COMMA namedParam )* RPAREN)? ;
frame               : formalQubit+ STRING ;

// built-in waveform types include: "flat", "gaussian", "draggaussian", "erfsquare"


////////////////////
// LEXER
////////////////////

keyword             : DEFGATE | DEFCIRCUIT | MEASURE | LABEL | HALT | JUMP | JUMPWHEN | JUMPUNLESS
                    | RESET | WAIT | NOP | INCLUDE | PRAGMA | DECLARE | SHARING | OFFSET | AS | MATRIX
                    | PERMUTATION | NEG | NOT | TRUE | FALSE | AND | IOR | XOR | OR | ADD | SUB | MUL
                    | DIV | MOVE | EXCHANGE | CONVERT | EQ | GT | GE | LT | LE | LOAD | STORE | PI | I
                    | SIN | COS | SQRT | EXP | CIS | CAPTURE | DEFCAL | DEFFRAME | DEFWAVEFORM
                    | DELAY | DIRECTION | FENCE | INITIALFREQUENCY | NONBLOCKING | PULSE | SAMPLERATE
                    | SETFREQUENCY | SETPHASE | SETSCALE | SHIFTPHASE | SWAPPHASE | RAWCAPTURE
                    | CONTROLLED | DAGGER | FORKED ;

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

AS                  : 'AS' ;
MATRIX              : 'MATRIX' ;
PERMUTATION         : 'PERMUTATION' ;
PAULISUM            : 'PAULI-SUM';

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

SIN                 : 'SIN' ;
COS                 : 'COS' ;
SQRT                : 'SQRT' ;
EXP                 : 'EXP' ;
CIS                 : 'CIS' ;

// Operators

PLUS                : '+' ;
MINUS               : '-' ;
TIMES               : '*' ;
DIVIDE              : '/' ;
POWER               : '^' ;

// analog keywords

CAPTURE             : 'CAPTURE' ;
DEFCAL              : 'DEFCAL' ;
DEFFRAME            : 'DEFFRAME' ;
DEFWAVEFORM         : 'DEFWAVEFORM' ;
DELAY               : 'DELAY' ;
DIRECTION           : 'DIRECTION' ;
FENCE               : 'FENCE' ;
INITIALFREQUENCY    : 'INITIAL-FREQUENCY' ;
NONBLOCKING         : 'NONBLOCKING' ;
PULSE               : 'PULSE' ;
SAMPLERATE          : 'SAMPLE-RATE' ;
SETFREQUENCY        : 'SET-FREQUENCY' ;
SETPHASE            : 'SET-PHASE' ;
SETSCALE            : 'SET-SCALE' ;
SHIFTPHASE          : 'SHIFT-PHASE' ;
SWAPPHASE           : 'SWAP-PHASE' ;
RAWCAPTURE          : 'RAW-CAPTURE' ;

// Modifiers

CONTROLLED          : 'CONTROLLED' ;
DAGGER              : 'DAGGER' ;
FORKED              : 'FORKED' ;

// Identifiers

IDENTIFIER          : ( ( [A-Za-z_] ) | ( [A-Za-z_] [A-Za-z0-9\-_]* [A-Za-z0-9_] ) ) ;

// Numbers

INT                 : [0-9]+ ;
FLOAT               : [0-9]+ ( '.' [0-9]+ )? ( ( 'e'|'E' ) ( '+' | '-' )? [0-9]+ )? ;

// String

STRING              : '"' ~( '\n' | '\r' | '"' )* '"';

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
NEWLINE             : (' ' | '\t' )* ( '\r'? '\n' | '\r' )+ ;

// Skips

COMMENT             : (' ' | '\t' )* '#' ~( '\n' | '\r' )* -> skip ;
SPACE               : ' ' -> skip ;

// Error

INVALID             : . ;
