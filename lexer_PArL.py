from enum import Enum, auto

# TokenType: Enumerates every possible token type the lexer can identify.
class TokenType(Enum):
    # Basic types
    identifier          = auto()
    integerType         = auto()
    floatType           = auto()
    boolean             = auto()
    colour              = auto()
    datatype            = auto()
    void                = auto()
    whitespace          = auto()

    # Operators and punctuation
    equals              = auto()  
    plus                = auto()  
    minus               = auto()  
    star                = auto()  
    slash               = auto() 
    operator            = auto()  
    relOperator         = auto()  
    decimalPoint        = auto()  
    semicolon           = auto()  
    comma               = auto()  
    arrow               = auto()  
    arrayLength         = auto()  

    # Brackets
    curvedBracketOpen   = auto()  
    curvedBracketClosed = auto() 
    curlyBracOpen       = auto() 
    curlyBracClosed     = auto()  
    squareBracketOpen   = auto()  
    squareBracketClosed = auto()  

    # Variable declarations
    variableDeclaration = auto()  
    variableDeclColon   = auto()  

    # Keywords
    if_keyword          = auto()
    else_keyword        = auto()
    let_keyword         = auto()
    fun_keyword         = auto()
    return_keyword      = auto()
    as_keyword          = auto()
    and_keyword         = auto()
    or_keyword          = auto()
    not_keyword         = auto()
    for_keyword         = auto()
    while_keyword       = auto()

    # Special commands
    print_keyword       = auto()
    write_keyword       = auto()
    write_box_keyword   = auto()
    read_keyword        = auto()
    delay_keyword       = auto()
    clear_keyword       = auto()
    random_keyword      = auto()
    randi_keyword       = auto()
    width_keyword       = auto()
    height_keyword      = auto()

    # Comments
    commentOpen         = auto()  
    commentClosed       = auto() 

    # End of input
    end                 = auto()

# Token: Holds information about a recognised token, including error info if needed
class Token:
    def __init__(self, t, l, line=1, col=1, error_message=None):
        self.type = t                  # TokenType
        self.lexeme = l                # The actual string matched
        self.line = line               # Line number (1-based)
        self.col = col                 # Column number (1-based)
        self.error_message = error_message  # Optional error message

# Lexer: Main class for tokenising input source code using a DFA (transition table)
class Lexer:
    def __init__(self):
        self.max_states = 30
        self.error_tokens = []  # List to keep track of error tokens (for reporting)

        # List of categories for DFA transitions (order matters!)
        self.lexeme_list = [
            "letter",            # a-z, A-Z
            "digit",             # 0-9
            "_",                 # underscore
            "ws",                # whitespace
            "eq",                # =
            "addOp",             # +, -
            "mulOp",             # *, /, %
            "dp",                # .
            "rop",               # <, >, !
            "sc",                # ;
            "cma",               # ,
            "vardecl",           # :
            "curvedBracketOpen", # (
            "curvedBracetClosed",# )
            "curlyBracketOpen",  # {
            "curlyBracketClosed",# }
            "squareBracketOpen", # [
            "squareBracketClosed",# ]
            "retArrow",          # ->
            "array",             # arrays (reserved)
            "print",             # print keyword
            "write",             # write keyword
            "read",              # read keyword
            "delay",             # delay keyword
            "width",             # width keyword
            "height",            # height keyword
            "colourVar",         # # (for colours)
            "colour",            # colour keyword
            "boolVar",           # boolean literals/vars
            "comment1",          # //
            "comment2",          # /*, */
            "other"              # catch-all for illegal/unmatched chars
        ]

        # List of states for DFA
        self.states_list = [n for n in range(self.max_states+1)]
        self.states_accp = [n for n in range(1, self.max_states+1)]  # Accepting states are 1..max_states

        # Reserved keywords mapping to their token types
        self.keywords = {
            "and": TokenType.and_keyword,
            "as": TokenType.as_keyword,
            "else": TokenType.else_keyword,
            "for": TokenType.for_keyword,
            "fun": TokenType.fun_keyword,
            "if": TokenType.if_keyword,
            "let": TokenType.let_keyword,
            "not": TokenType.not_keyword,
            "or": TokenType.or_keyword,
            "return": TokenType.return_keyword,
            "while": TokenType.while_keyword,
            "__height": TokenType.height_keyword,
            "__width": TokenType.width_keyword
        }

        # Pad-specific commands
        self.padKeywords = {
            "__clear": TokenType.clear_keyword,
            "__delay": TokenType.delay_keyword,
            "__print": TokenType.print_keyword,
            "__randi": TokenType.randi_keyword,
            "__random_int": TokenType.random_keyword,
            "__read": TokenType.read_keyword,
            "__write": TokenType.write_keyword,
            "__write_box": TokenType.write_box_keyword
        }

        # Special sequences
        self.returnArrow = {
            "->": TokenType.arrow
        }

        # Boolean literal values
        self.boolVals = {
            "true": TokenType.boolean,
            "false": TokenType.boolean
        }

        # Supported datatype words
        self.dataTypes = {
            "int": TokenType.datatype,
            "float": TokenType.datatype,
            "bool": TokenType.datatype,
            "colour": TokenType.datatype
        }

        # Supported comment formats
        self.supportedComments = {
            "//": TokenType.commentOpen,
            "/*": TokenType.commentOpen,
            "*/": TokenType.commentClosed
        }

        # DFA table dimensions
        self.rows = len(self.states_list)
        self.cols = len(self.lexeme_list)
        self.transition_table = [[-1 for j in range(self.cols)] for i in range(self.rows)]

        self.InitialiseTransitionTable()  # Sets up DFA transitions

    # Sets up the DFA's transitions for each state and character category
    def InitialiseTransitionTable(self):
        # States for identifiers (letters and underscores)
        self.transition_table[0][self.lexeme_list.index("letter")] = 1
        self.transition_table[0][self.lexeme_list.index("_")] = 1
        self.transition_table[1][self.lexeme_list.index("_")] = 1
        self.transition_table[1][self.lexeme_list.index("letter")] = 1
        self.transition_table[1][self.lexeme_list.index("digit")] = 1

        # Whitespace transitions
        self.transition_table[0][self.lexeme_list.index("ws")] = 2
        self.transition_table[2][self.lexeme_list.index("ws")] = 2

        # Punctuation & operator transitions
        self.transition_table[0][self.lexeme_list.index("eq")] = 3        
        self.transition_table[0][self.lexeme_list.index("cma")] = 4
        self.transition_table[0][self.lexeme_list.index("digit")] = 5        
        self.transition_table[5][self.lexeme_list.index("digit")] = 5
        self.transition_table[7][self.lexeme_list.index("addOp")] = 5   

        self.transition_table[0][self.lexeme_list.index("sc")] = 6

        self.transition_table[0][self.lexeme_list.index("addOp")] = 7
        self.transition_table[7][self.lexeme_list.index("addOp")] = 7
        self.transition_table[0][self.lexeme_list.index("mulOp")] = 7
        self.transition_table[7][self.lexeme_list.index("mulOp")] = 7

        self.transition_table[0][self.lexeme_list.index("dp")] = 8
        self.transition_table[8][self.lexeme_list.index("dp")] = 8
        self.transition_table[5][self.lexeme_list.index("dp")] = 8

        self.transition_table[0][self.lexeme_list.index("rop")] = 9
        self.transition_table[3][self.lexeme_list.index("eq")] = 9  
        self.transition_table[9][self.lexeme_list.index("eq")] = 9  
        self.transition_table[7][self.lexeme_list.index("rop")] = 9  
        self.transition_table[4][self.lexeme_list.index("eq")] = 9

        self.transition_table[5][self.lexeme_list.index("dp")] = 8  
        self.transition_table[5][self.lexeme_list.index("digit")] = 5  
        self.transition_table[8][self.lexeme_list.index("digit")] = 10  
        self.transition_table[10][self.lexeme_list.index("digit")] = 10 

        self.transition_table[0][self.lexeme_list.index("colourVar")] = 12
        self.transition_table[0][self.lexeme_list.index("colour")] = 12
        self.transition_table[12][self.lexeme_list.index("digit")] = 12
        self.transition_table[12][self.lexeme_list.index("letter")] = 12

        self.transition_table[0][self.lexeme_list.index("boolVar")] = 13

        self.transition_table[0][self.lexeme_list.index("vardecl")] = 14

        self.transition_table[0][self.lexeme_list.index("curvedBracketOpen")] = 15
        self.transition_table[0][self.lexeme_list.index("curvedBracetClosed")] = 17

        self.transition_table[0][self.lexeme_list.index("curlyBracketOpen")] = 16
        self.transition_table[0][self.lexeme_list.index("curlyBracketClosed")] = 18

        self.transition_table[0][self.lexeme_list.index("print")] = 19
        self.transition_table[0][self.lexeme_list.index("write")] = 20
        self.transition_table[0][self.lexeme_list.index("read")] = 21
        self.transition_table[0][self.lexeme_list.index("delay")] = 22

        self.transition_table[0][self.lexeme_list.index("squareBracketOpen")] = 24
        self.transition_table[0][self.lexeme_list.index("squareBracketClosed")] = 25

    # Checks if a state is an accepting state
    def AcceptingStates(self, state):
        try:
            self.states_accp.index(state)
            return True
        except ValueError:
            return False
    
    # Verifies if a colour literal is in the form "#RRGGBB"
    def ValidateColour(self, color):
        if len(color) == 7 and color[0] == '#' and all(c in '0123456789abcdefABCDEF' for c in color[1:]):
            return True
        else:
            return False

    # Determines the token type for the final DFA state
    def GetTokenTypeByFinalState(self, state, lexeme):
        if state == 1:
            if lexeme in self.keywords:
                return Token(self.keywords[lexeme], lexeme)
            elif lexeme in self.padKeywords:
                return Token(self.padKeywords[lexeme], lexeme)
            elif lexeme in self.dataTypes:
                return Token(TokenType.datatype, lexeme)
            elif lexeme in self.boolVals:
                return Token(TokenType.boolean, lexeme)
            else:
                return Token(TokenType.identifier, lexeme)
        elif state == 2:
            return Token(TokenType.whitespace, lexeme)
        elif state == 3:
            return Token(TokenType.equals, lexeme)
        elif state == 4:
            return Token(TokenType.comma, lexeme)
        elif state == 5  or state == 11:
            return Token(TokenType.integerType, lexeme)
        elif state == 6:
            return Token(TokenType.semicolon, lexeme)
        elif state == 7:
            if lexeme in self.supportedComments:
                return Token(self.supportedComments[lexeme], lexeme)
            else:
                return Token(TokenType.operator, lexeme)
        elif state == 8:
            return Token(TokenType.decimalPoint, lexeme)
        elif state == 9:
            if lexeme in self.returnArrow:
                return Token(TokenType.arrow, lexeme)
            elif lexeme in ("!=", "==", "<", ">", "<=", ">="):
                return Token(TokenType.relOperator, lexeme)
            else:
                return Token(TokenType.relOperator, lexeme)
        elif state == 10:
            return Token(TokenType.floatType, lexeme)  
        elif state == 12:
            if lexeme.startswith('#') and self.ValidateColour(lexeme):
                return Token(TokenType.colour, lexeme)
            else:
                return Token(TokenType.void, lexeme, error_message="Invalid colour literal")
        elif state == 14:
            return Token(TokenType.variableDeclaration, lexeme)
        elif state == 15:
            return Token(TokenType.curvedBracketOpen, lexeme)
        elif state == 16:
            return Token(TokenType.curlyBracOpen, lexeme)
        elif state == 17:
            return Token(TokenType.curvedBracketClosed, lexeme)
        elif state == 18:
            return Token(TokenType.curlyBracClosed, lexeme)
        elif state == 23:
            return Token(TokenType.arrayLength, lexeme)
        elif state == 24:
            return Token(TokenType.squareBracketOpen, lexeme)
        elif state == 25:
            return Token(TokenType.squareBracketClosed, lexeme)
        else:
            return 'default result'

    # Categorises a character for the DFA transition table
    def CatChar(self, character):
        cat = "other"
        if character.isalpha(): cat = "letter"
        elif character.isdigit(): cat = "digit"
        elif character == "_": cat = "_"
        elif character == " ": cat = "ws"
        elif character == ";": cat = "sc"
        elif character == ",": cat = "cma"
        elif character == "=": cat = "eq"
        elif character == "+" or character == "-": cat = "addOp"
        elif character == "*" or character == "/" or character == "%": cat = "mulOp"
        elif character == ".": cat = "dp"
        elif character == "<" or character == ">" or character == "!": cat = "rop"
        elif character == ":": cat = "vardecl"
        elif character == "#": cat = "colourVar"
        elif character == "(": cat = "curvedBracketOpen"
        elif character == ")": cat = "curvedBracetClosed"
        elif character == "{": cat = "curlyBracketOpen"
        elif character == "}": cat = "curlyBracketClosed"
        elif character == "[": cat = "squareBracketOpen"
        elif character == "]": cat = "squareBracketClosed"
        elif character == '\n' or character == '\r' or character == '\t':
            return "ws"
        else:
            return None
        return cat

    # Checks if the end of the input string has been reached
    def EndOfInput(self, src_program_str, src_program_idx):
        return src_program_idx > len(src_program_str)-1

    # Gets the next character if not at end of input
    def NextChar(self, src_program_str, src_program_idx):
        if not self.EndOfInput(src_program_str, src_program_idx):
            return True, src_program_str[src_program_idx]
        else: 
            return False, "."

    # The core DFA loop: returns a Token and error-tracking if applicable
    def NextToken(self, src_program_str, src_program_idx, line, col):
        state = 0
        lexeme = ""
        start_idx = src_program_idx
        start_line = line
        start_col = col
        newlines = 0
        last_line_col = col

        while True:
            exists, ch = self.NextChar(src_program_str, src_program_idx)
            if not exists:
                break

            # Track line and column positions for error reporting
            if ch == '\n':
                newlines += 1
                last_line_col = 1
                col = 1
                line += 1
            else:
                col += 1
                last_line_col = col

            cat = self.CatChar(ch)
            if cat is None:
                lexeme += ch
                error_msg = f"Illegal character '{ch}' encountered."
                token = Token(TokenType.void, lexeme, start_line, start_col, error_message=error_msg)
                self.error_tokens.append(token)  # Track error
                return token, lexeme, newlines, last_line_col

            next_state = self.transition_table[state][self.lexeme_list.index(cat)]
            if next_state == -1:
                break

            state = next_state
            lexeme += ch
            src_program_idx += 1

        # If non-accepting, try to backtrack to last valid token
        temp_state = state
        temp_lexeme = lexeme
        temp_newlines = newlines
        temp_last_col = last_line_col
        while not self.AcceptingStates(temp_state) and temp_lexeme:
            if temp_lexeme[-1] == '\n':
                temp_newlines -= 1
            temp_lexeme = temp_lexeme[:-1]
            src_program_idx -= 1
            temp_state = 0
            for c in temp_lexeme:
                temp_cat = self.CatChar(c)
                temp_state = self.transition_table[temp_state][self.lexeme_list.index(temp_cat)]

        # If no lexeme found, it's an error
        if not temp_lexeme:
            error_msg = f"Unknown or illegal token: '{lexeme}'"
            token = Token(TokenType.void, lexeme, start_line, start_col, error_message=error_msg)
            self.error_tokens.append(token)
            return token, lexeme, newlines, last_line_col

        # Get the final token type (if an error, record it)
        token = self.GetTokenTypeByFinalState(temp_state, temp_lexeme)
        if isinstance(token, str):
            error_msg = "Unknown token type encountered."
            token = Token(TokenType.void, temp_lexeme, start_line, start_col, error_message=error_msg)
            self.error_tokens.append(token)
        elif token.type == TokenType.void:
            token.line = start_line
            token.col = start_col
            if not token.error_message:
                token.error_message = "Invalid token encountered."
            self.error_tokens.append(token)
        return token, temp_lexeme, temp_newlines, temp_last_col

    # Generates the list of tokens for an input string (main lexer entry point)
    def GenerateTokens(self, src_program_str):
        print("INPUT:: " + src_program_str)
        tokens_list = []
        src_program_idx = 0
        line = 1
        col = 1

        while not self.EndOfInput(src_program_str, src_program_idx):
            # Handle single-line comments
            if src_program_str[src_program_idx:src_program_idx+2] == "//":
                next_nl = src_program_str.find('\n', src_program_idx)
                if next_nl == -1:
                    break  
                src_program_idx = next_nl + 1
                line += 1
                col = 1
                continue

            # Handle multi-line comments (block comments)
            if src_program_str[src_program_idx:src_program_idx+2] == "/*":
                end_idx = src_program_str.find("*/", src_program_idx+2)
                if end_idx == -1:
                    error_msg = f"Unterminated block comment starting at line {line}, col {col}"
                    token = Token(TokenType.void, src_program_str[src_program_idx:], line, col, error_message=error_msg)
                    self.error_tokens.append(token)  # Track error
                    print("ERROR:", error_msg)
                    break
                comment_content = src_program_str[src_program_idx+2:end_idx]
                line += comment_content.count('\n')
                last_newline = comment_content.rfind('\n')
                if last_newline != -1:
                    col = len(comment_content) - last_newline
                else:
                    col += (end_idx + 2) - src_program_idx
                src_program_idx = end_idx + 2
                continue

            # Use the DFA to get next token
            token, lexeme, newlines, last_line_col = self.NextToken(src_program_str, src_program_idx, line, col)
            token.line = line
            token.col = col

            # Ignore comment tokens in output, but update position
            if token.type in (TokenType.commentOpen, TokenType.commentClosed):
                src_program_idx += len(lexeme)
                if newlines > 0:
                    line += newlines
                    col = last_line_col
                else:
                    col += len(lexeme)
                continue

            tokens_list.append(token)

            if newlines > 0:
                line += newlines
                col = last_line_col
            else:
                col += len(lexeme)

            src_program_idx += len(lexeme)

            # If there was an error token, stop further lexing
            if token.type == TokenType.void:
                break

        # Add end-of-input token
        tokens_list.append(Token(TokenType.end, "END", line, col))
        return tokens_list

    # Utility: returns all pad keywords
    def GetPadKeywords(self):
        return self.padKeywords

    # Debug: print the DFA transition table
    def print_transition_table(self):
        print("Transition Table:")
        for row in self.transition_table:
            print(row)

# Example usage & test case 

# test_string_a = "let c:colour = #ff00g1;"

# toks = lex.GenerateTokens(test_string_a)

# # Print tokens with error messages where relevant
# for t in toks:
#     if t.type == TokenType.void and t.error_message:
#         print(f"ERROR: {t.error_message} at line {t.line}, col {t.col} (lexeme: '{t.lexeme}')")
#     else:
#         print(f"{t.type} {t.lexeme}")
