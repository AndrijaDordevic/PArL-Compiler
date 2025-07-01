from astnodes_PArL import *
from visitor_print_PArL import *
from semantic_analyser_PArL import SemanticAnalyser
from code_generator_PArL import IRCodeGenerator
from lexer_PArL import *
import random
import builtins

# Custom exception for syntax errors in the parser
class ParseError(builtins.SyntaxError):
    # Raised when the lexer/parser encounters invalid syntax.

    def __init__(self, message: str, token=None):
        self.token = token

        lineno  = getattr(token, "line", None)
        offset  = getattr(token, "col",  None)   # ← use .col here, not .column

        super().__init__(message, ("<source>", lineno, offset, ""))

    def __str__(self):
        if self.token:
            # Use the Token’s .line and .col fields:
            line   = getattr(self.token, "line", "?")
            column = getattr(self.token, "col",  "?")
            lexeme = getattr(self.token, "lexeme", "")
            return (f"Syntax error at line {line}, col {column} near “{lexeme}”: {self.args[0]}")
        return f"Syntax error: {self.args[0]}"

#  Runtime switch 
# If set to True *before* parsing, the helper _raise_syntax() will raise the
# built-in SyntaxError
RAISE_BUILTIN_SYNTAX_ERROR = False


def _raise_syntax(message: str, token=None):
    """
    Central helper used throughout the parser instead of `raise ParseError(...)`.

    It honours the global RAISE_BUILTIN_SYNTAX_ERROR toggle so that callers can
    choose between receiving a rich ParseError (default) or the plain built-in
    SyntaxError (for maximum external compatibility).
    """
    if RAISE_BUILTIN_SYNTAX_ERROR:
        lineno  = getattr(token, "line",   None)
        offset  = getattr(token, "column", None)
        raise builtins.SyntaxError(message, ("<source>", lineno, offset, ""))
    else:
        raise ParseError(message, token)

# Main Parser class
class Parser:
    def __init__(self, src_program_str):
        self.name = "PARSER" # Name for debugging
        self.lexer = Lexer() # Instance of the lexer to get tokens
        self.index = -1  # Index for tracking current token
        self.next = -1   # Index for peeking ahead
        self.if_count = 0 # Count of if-statements, for internal use
        self.let_count = 0 # Count of let-statements, for internal use
        self.stmt_type = [] # List of parsed statements (as AST nodes)
        self.declared_vars = {} # Track variable declarations: name -> type
        self.declared_var_vals = {} # Track variable values: name -> value
        self.declared_functions = {} # Track functions: name -> definition
        self.declared_functions_params = {} # Track function parameters
        self.declared_arrays = set() # Track declared arrays
        self.src_program = src_program_str # Store the source code
        self.stmt_list = self.src_program.split(";") # Split source into statements
        self.stmt_index = 0 # Statement index for parsing multiple statements
        self.tokens = self.lexer.GenerateTokens(self.src_program) # Tokenise input
        self.crtToken = Token("", TokenType.void) # Current token
        self.nextToken = Token("", TokenType.void) # Next token (lookahead)
        self.ASTroot = ASTProgramNode # Root of the AST

    # Move to next token, skipping whitespace tokens
    def GetNextTokenSkipWS(self):
        self.index += 1  
        if (self.index < len(self.tokens)):
            self.crtToken = self.tokens[self.index]
        else:
            self.crtToken = Token(TokenType.end, "END") # End of input
    
    # Peek at next token without advancing the main token pointer
    def PredictNextTokenSkipWS(self):
        if (self.next < len(self.tokens)):
            self.nextToken = self.tokens[self.next]
        else:
            self.nextToken = Token(TokenType.end, "END")
    
    # Consume a comma if present (used for parsing argument lists)
    def _eat_comma(self):
        if self.crtToken.type == TokenType.comma:
            self.GetNextToken()
    
    # Parse a print statement: print(expression)
    def _parse_print(self):
        self.GetNextToken()
        expr = self.ParseExpression()
        node = ASTPrintNode(expr)
        self.stmt_type.append(node)
        return node

    # Parse a delay statement: delay(expression)
    def _parse_delay(self):
        self.GetNextToken()
        expr = self.ParseExpression()
        node = ASTDelayNode(expr)
        self.stmt_type.append(node)
        return node

    # Parse a clear statement: clear(expression)
    def _parse_clear(self):
        self.GetNextToken()
        expr = self.ParseExpression()
        node = ASTClearNode(expr)
        self.stmt_type.append(node)
        return node

    # Parse a write statement: write(x, y, colour)
    def _parse_write(self):
        self.GetNextToken()
        x      = self.ParseExpression()
        self._eat_comma()
        y      = self.ParseExpression()
        self._eat_comma()
        colour = self.ParseExpression()
        node = ASTWriteNode(colour, y, x)
        self.stmt_type.append(node)
        return node

    # Parse a write_box statement: write_box(x, y, w, h, colour)
    def _parse_write_box(self):
        self.GetNextToken()
        x      = self.ParseExpression()
        self._eat_comma()
        y      = self.ParseExpression()
        self._eat_comma()
        w      = self.ParseExpression()
        self._eat_comma()
        h      = self.ParseExpression()
        self._eat_comma()
        colour = self.ParseExpression()
        node = ASTWriteBoxNode(x, y, w, h, colour)
        self.stmt_type.append(node)
        return node

    # Parse variable declaration with a random value (e.g., let x: int = random(10))
    def _parse_var_decl_with_random(self):
        pad_type = pad_val = None
        have_to_decl = False
        if self.crtToken.type == TokenType.let_keyword:
            have_to_decl = True
            self.GetNextToken()
        if self.crtToken.type == TokenType.identifier:
            var_name = ASTVariableNode(self.crtToken.lexeme)
            self.GetNextToken()
        else:
            var_name = None
        if self.crtToken.type == TokenType.variableDeclaration:
            self.GetNextToken()
        if self.crtToken.type == TokenType.datatype:
            var_type = ASTDataTypeNode(self.crtToken.lexeme)
            self.GetNextToken()
        else:
            var_type = None

        # Register the variable if it is a declaration
        if have_to_decl and var_name and var_type:
            self.declared_vars[var_name.lexeme] = var_type.type

        if self.crtToken.type == TokenType.equals:
            self.GetNextToken()
        if self.crtToken.type == TokenType.curvedBracketOpen :
            self.GetNextToken()
        if self.crtToken.type == TokenType.random_keyword:
            pad_type = self.crtToken.type
            self.GetNextToken()
            pad_val = self.ParseExpression()
            if have_to_decl:
                # If variable is not width/height, random int up to value, else max 35
                if not (isinstance(pad_val, ASTWidthNode) or isinstance(pad_val, ASTHeightNode)):
                    self.declared_var_vals[var_name.lexeme] = random.randint(0, int(pad_val.value))
                else:
                    self.declared_var_vals[var_name.lexeme] = random.randint(0, 35)
        if self.crtToken.type == TokenType.curvedBracketClosed:
            self.GetNextToken()

        if pad_type == TokenType.random_keyword:
            node = ASTVarDeclNode(var_name, var_type, ASTRandomNode(pad_val))
            self.stmt_type.append(node)
            return node

        return None

    # Move to next non-whitespace token
    def GetNextToken(self):
        self.GetNextTokenSkipWS()
        while (self.crtToken.type == TokenType.whitespace):
            self.GetNextTokenSkipWS()
    
    # Look ahead to next non-whitespace token (doesn't advance the main pointer)
    def PredictNextToken(self):
        self.next = self.index + 1   
        self.PredictNextTokenSkipWS()
        while (self.nextToken.type == TokenType.whitespace):
            self.PredictNextTokenSkipWS()
            self.next += 1

    # Parse a primary expression (literals, identifiers, array, etc)
    def ParsePrimary(self):
        if self.crtToken.type == TokenType.squareBracketOpen:
            # Array literal
            self.GetNextToken()
            elements = []
            while True:
                elements.append(self.ParseExpression())
                if self.crtToken.type == TokenType.comma:
                    self.GetNextToken()
                    continue
                break
            self.Expect(TokenType.squareBracketClosed)
            self.GetNextToken()
            node = ASTArrayNode(elements, ASTIntegerNode(len(elements)))

        elif self.crtToken.type == TokenType.integerType:
            # Integer literal
            node = ASTIntegerNode(self.crtToken.lexeme)
            self.GetNextToken()

        elif self.crtToken.type == TokenType.floatType:
            # Float literal
            node = ASTFloatNode(self.crtToken.lexeme)
            self.GetNextToken()

        elif self.crtToken.type == TokenType.boolean:
            # Boolean literal
            val = 1 if self.crtToken.lexeme == "true" else 0
            node = ASTBoolNode(val)
            self.GetNextToken()

        elif self.crtToken.type == TokenType.colour:
            # Colour literal
            node = ASTColourNode(self.crtToken.lexeme)
            self.GetNextToken()

        elif self.crtToken.type == TokenType.read_keyword:   
            # Read statement: read(x, y)
            self.GetNextToken()                              
            left = self.ParseExpression()
            self.Expect(TokenType.comma)
            self.GetNextToken()                                
            right = self.ParseExpression()
            node = ASTReadNode(left, right)
        
        elif self.crtToken.type == TokenType.random_keyword:
            # random(value)
            self.GetNextToken()                
            if self.crtToken.type == TokenType.curvedBracketOpen:
                self.GetNextToken()             
                inside = self.ParseExpression()
                self.Expect(TokenType.curvedBracketClosed)
                self.GetNextToken()             
            else:
                inside = self.ParseExpression() 
            node = ASTRandomNode(inside)

        elif self.crtToken.type == TokenType.randi_keyword:
            # randi(value)
            self.GetNextToken() 
            if self.crtToken.type == TokenType.curvedBracketOpen:
                self.GetNextToken()  
                arg_expr = self.ParseExpression()
                self.Expect(TokenType.curvedBracketClosed)
                self.GetNextToken()  
            else:
                arg_expr = self.ParseExpression()  
            node = ASTRandomNode(arg_expr)

        elif self.crtToken.type == TokenType.width_keyword:
            # width keyword
            node = ASTWidthNode(self.crtToken.lexeme)
            self.GetNextToken()

        elif self.crtToken.type == TokenType.height_keyword:
            # height keyword
            node = ASTHeightNode(self.crtToken.lexeme)
            self.GetNextToken()

        elif self.crtToken.type == TokenType.identifier:
            # Identifier could be a variable, array element, or function call
            self.PredictNextToken()
            if self.nextToken.type == TokenType.squareBracketOpen:
                # Array element access: varname[expr]
                node = self.ParseArrayElement()

            elif self.nextToken.type == TokenType.curvedBracketOpen:
                # Function call: func(expr1, expr2, ...)
                func_name = ASTVariableNode(self.crtToken.lexeme)
                self.GetNextToken()  # consume function name
                self.GetNextToken()  # consume '('

                args = []
                if self.crtToken.type != TokenType.curvedBracketClosed:
                    while True:
                        args.append(self.ParseExpression())
                        if self.crtToken.type == TokenType.comma:
                            self.GetNextToken()
                            continue
                        break
                self.Expect(TokenType.curvedBracketClosed)
                self.GetNextToken()
                node = ASTFunctionCallExprNode(func_name, args)

            else:
                # Simple variable access
                node = ASTVariableNode(self.crtToken.lexeme)
                self.GetNextToken()

        elif self.crtToken.type == TokenType.curvedBracketOpen:
            # Parenthesised expression (just grouping)
            self.GetNextToken()
            node = self.ParseExpression()
            self.Expect(TokenType.curvedBracketClosed)
            self.GetNextToken()

        else:
            _raise_syntax(f"unexpected token {self.crtToken.type}", self.crtToken)

        # Optional: Cast expression using 'as'
        while self.crtToken.type == TokenType.as_keyword:
            self.GetNextToken() 
            if self.crtToken.type != TokenType.datatype:
                _raise_syntax("expected datatype after 'as'", self.crtToken)
            target_type = ASTDataTypeNode(self.crtToken.lexeme)
            self.GetNextToken()
            node = ASTCastNode(node, target_type)
            # If casting integer to colour, convert value to hex
            if (target_type.type == "colour"
                and isinstance(node.expr, ASTIntegerNode)):
                hex_str = f"#{int(node.expr.value):06x}"
                node = ASTColourNode(hex_str)

        return node

    # Parse unary operations (negation, not, etc)
    def ParseUnary(self):
        if self.crtToken.type == TokenType.operator and self.crtToken.lexeme == "-":
            op = self.crtToken.lexeme
            self.GetNextToken()
            operand = self.ParseUnary()     
            return ASTUnaryOpNode(op, operand)

        if self.crtToken.type == TokenType.not_keyword:
            op = "not"
            self.GetNextToken()
            operand = self.ParseUnary()
            return ASTUnaryOpNode(op, operand)

        return self.ParsePrimary()
    
    # Parse multiplication, division, modulus (left-associative)
    def ParseMul(self):
        node = self.ParseUnary()
        while self.crtToken.type in (
                TokenType.operator,          # *, /, %
                TokenType.and_keyword        
        ):
            # only *, /, and % should stay in this loop; anything else breaks out
            if self.crtToken.type == TokenType.operator \
            and self.crtToken.lexeme not in ("*", "/", "%"):
                break

            # At this point, lexeme is either "*", "/" or "%"
            op = self.crtToken.lexeme if self.crtToken.type == TokenType.operator else "and"
            self.GetNextToken()
            rhs = self.ParseUnary()
            node = ASTBinaryOpNode(op, node, rhs)
        return node
    
    # Parse addition and subtraction (left-associative)
    def ParseAdd(self):
        node = self.ParseMul()
        while True:
            tok = self.crtToken
            if tok.type == TokenType.operator and tok.lexeme in ("+", "-"):
                op = tok.lexeme
            elif tok.type == TokenType.or_keyword:
                op = "or"
            else:
                break
            self.GetNextToken()
            rhs = self.ParseMul()
            node = ASTBinaryOpNode(op, node, rhs)
        return node
    
    # Parse relational operators (==, <, >, etc)
    def ParseRel(self):
        node = self.ParseAdd()
        while (
            self.crtToken.type == TokenType.relOperator
            or (
                self.crtToken.type == TokenType.operator
                and self.crtToken.lexeme in {"==", "!=", "<", "<=", ">", ">="}
            )
        ):
            op_tok = self.crtToken
            self.GetNextToken()
            rhs = self.ParseAdd()
            node = ASTBinaryOpNode(op_tok.lexeme, node, rhs)

        while self.crtToken.type == TokenType.as_keyword:
            self.GetNextToken()
            self.Expect(TokenType.datatype)
            target = ASTDataTypeNode(self.crtToken.lexeme)
            self.GetNextToken()
            node = ASTCastNode(node, target)
            if (target.type == "colour" and isinstance(node.expr, ASTIntegerNode)):
                hex_str = f"#{int(node.expr.value):06x}"
                node = ASTColourNode(hex_str)

        return node

    # Parse a full expression (lowest precedence is boolean)
    def ParseExpression(self):
        return self.ParseRel()

    # Utility: Raise error if current token is not the expected type
    def Expect(self, tok_type):
        if self.crtToken.type != tok_type:
            _raise_syntax(f"expected {tok_type} but found {self.crtToken.type}", self.crtToken)
    
    # Parse end-of-program marker
    def ParseEndToken(self):
        if (self.crtToken.type == TokenType.end):
            self.stmt_type.append(ASTEndNode())
            return ASTEndNode()
    
    # Parse a while loop statement
    def ParseWhileStatement(self):
        self.Expect(TokenType.while_keyword)
        self.GetNextToken()

        self.Expect(TokenType.curvedBracketOpen  )
        self.GetNextToken()

        cond_expr = self.ParseExpression()

        self.Expect(TokenType.curvedBracketClosed)
        self.GetNextToken()

        self.Expect(TokenType.curlyBracOpen)
        self.GetNextToken()

        body = self.ParseBlock()
        self.Expect(TokenType.curlyBracClosed)
        self.GetNextToken()

        node = ASTWhileNode(ASTConditionNode(cond_expr), body)
        self.stmt_type.append(node)
        return node

    # Parse a return statement
    def ParseReturn(self):
        self.GetNextToken()
        ret_val = self.ParseExpression()
        if isinstance(ret_val, (ASTArrayNode, ASTVariableNode)) and \
        (isinstance(ret_val, ASTArrayNode) or ret_val.lexeme in self.declared_arrays):
            node = ASTReturnArrayNode(ret_val)
        else:
            node = ASTReturnNode(ret_val)
        self.stmt_type.append(node)
        return node
    
    # Parse array element access: arr[index]
    def ParseArrayElement(self):
            arr_var = ASTVariableNode(self.crtToken.lexeme)
            self.GetNextToken()                     
            self.Expect(TokenType.squareBracketOpen)
            self.GetNextToken()                    

            index = self.ParseExpression()

            self.Expect(TokenType.squareBracketClosed)
            self.GetNextToken()                     

            return ASTArrayElementNode(index, arr_var)
        
    # Parse any of the special PAD functions, using a dispatch table for keywords
    def ParsePadFunction(self):
        """
        Handles all PAD keywords:
            __print(expr)
            __delay(expr)
            __clear(expr)
            __write(x, y, colour)
            __write_box(x, y, w, h, colour)
        """

        pad_dispatch = {
            TokenType.print_keyword     : self._parse_print,
            TokenType.delay_keyword     : self._parse_delay,
            TokenType.clear_keyword     : self._parse_clear,
            TokenType.write_keyword     : self._parse_write,
            TokenType.write_box_keyword : self._parse_write_box,
        }

        token_type = self.crtToken.type

        #  Fast path 
        if token_type in pad_dispatch:
            return pad_dispatch[token_type]()          # call the helper

        #  Fallback:  let x : int = random(…) 
        return self._parse_var_decl_with_random()

    # Parse an assignment statement, e.g., x = expr or arr[2] = expr
    def ParseAssignment(self):
        # The statement must start with an identifier
        if self.crtToken.type != TokenType.identifier:
            _raise_syntax(f"expected identifier, got {self.crtToken.type}", self.crtToken)

        # Peek ahead to check if this is an array element assignment
        self.PredictNextToken()
        if self.nextToken.type == TokenType.squareBracketOpen:
            lhs_node = self.ParseArrayElement()     # Parse arr[expr] as LHS
        else:
            lhs_node = ASTVariableNode(self.crtToken.lexeme)  # Just a variable
            self.GetNextToken()                     

        self.Expect(TokenType.equals)  # Expect '='
        self.GetNextToken()                         

        rhs_node = self.ParseExpression()  # Parse the right-hand side

        assign_node = ASTAssignmentNode(lhs_node, rhs_node)
        self.stmt_type.append(assign_node)
        return assign_node

    # Parse an assignment to a specific element of an array, e.g. arr[1] = expr
    def ParseAssignmentArray(self):
        # Must start with an identifier (the array name)
        if self.crtToken.type != TokenType.identifier:
            _raise_syntax(f"expected array name, got {self.crtToken.type}", self.crtToken)

        arr_var_node = ASTVariableNode(self.crtToken.lexeme)  # The array variable
        self.GetNextToken()

        self.Expect(TokenType.squareBracketOpen)  # Expect '['
        self.GetNextToken()

        index_node = self.ParseExpression()       # Parse the index inside [ ]

        self.Expect(TokenType.squareBracketClosed)  # Expect ']'
        self.GetNextToken()

        element_node = ASTArrayElementNode(index_node, arr_var_node)  # arr[index] node

        self.Expect(TokenType.equals)  # Expect '='
        self.GetNextToken()

        rhs_node = self.ParseExpression()  # Parse right-hand side expression

        assign_array = ASTAssignmentNode(element_node, rhs_node)
        self.stmt_type.append(assign_array)
        return assign_array

    # Parse a variable declaration, e.g., let x: int = expr;
    def ParseVarDecl(self):
        if self.crtToken.type == TokenType.let_keyword:
            self.GetNextToken()  # Skip 'let'

        # Next must be an identifier (variable name)
        if self.crtToken.type != TokenType.identifier:
            _raise_syntax(f"expected identifier, got {self.crtToken.type}", self.crtToken)

        var_name = ASTVariableNode(self.crtToken.lexeme)
        self.GetNextToken()

        self.Expect(TokenType.variableDeclaration)  # Usually ':'
        self.GetNextToken()

        if self.crtToken.type != TokenType.datatype:
            _raise_syntax(f"expected datatype, got {self.crtToken.type}", self.crtToken)

        var_type = ASTDataTypeNode(self.crtToken.lexeme)
        self.GetNextToken()

        # Register the variable type in the parser's context
        self.declared_vars[var_name.lexeme] = var_type.type

        # Optionally parse initial value if '=' present
        if self.crtToken.type == TokenType.equals:
            self.GetNextToken()
            init_expr = self.ParseExpression()
        else:
            init_expr = None

        decl_node = ASTVarDeclNode(var_name, var_type, init_expr)
        self.stmt_type.append(decl_node)
        return decl_node

    # Parse an array variable declaration, e.g., let arr: int[5] = [1,2,3,4,5];
    def ParseVarDeclArray(self):

        # 1) “let”
        if self.crtToken.type == TokenType.let_keyword:
            self.GetNextToken()                                     # skip 'let'

        # 2) identifier
        if self.crtToken.type != TokenType.identifier:
            _raise_syntax("expected identifier", self.crtToken)
        var_name = ASTVariableNode(self.crtToken.lexeme)
        self.GetNextToken()

        # 3) “:”
        self.Expect(TokenType.variableDeclaration)
        self.GetNextToken()

        # 4) datatype
        if self.crtToken.type != TokenType.datatype:
            _raise_syntax("expected datatype", self.crtToken)
        var_type = ASTDataTypeNode(self.crtToken.lexeme)
        self.GetNextToken()

        # 5) “[”  <len>?  “]”
        self.Expect(TokenType.squareBracketOpen)
        self.GetNextToken()

        length_node = None
        if self.crtToken.type == TokenType.integerType:
            length_node = ASTIntegerNode(int(self.crtToken.lexeme))
            self.GetNextToken()

        self.Expect(TokenType.squareBracketClosed)
        self.GetNextToken()

        self.Expect(TokenType.equals)
        self.GetNextToken()

        init_expr = self.ParseExpression()

        if not isinstance(
            init_expr,
            (ASTArrayNode,            
             ASTFunctionCallExprNode, 
             ASTVariableNode)):       
            _raise_syntax("initializer must be an array value", self.crtToken)

        if isinstance(init_expr, ASTArrayNode):
            init_expr.length_hint = length_node

        decl_node = ASTVarDeclArrayNode(var_name, var_type, init_expr)

        decl_node.declared_len = length_node

        # bookkeeping
        self.declared_vars[var_name.lexeme] = var_type.type
        self.declared_arrays.add(var_name.lexeme)
        self.stmt_type.append(decl_node)
        return decl_node

    # Parse an if or if-else statement
    def ParseIfStatement(self):
        self.Expect(TokenType.if_keyword)
        self.GetNextToken()

        self.Expect(TokenType.curvedBracketOpen  )  # '('
        self.GetNextToken()

        cond_expr = self.ParseExpression()  # Parse condition

        self.Expect(TokenType.curvedBracketClosed)  # ')'
        self.GetNextToken()

        self.Expect(TokenType.curlyBracOpen)  # '{'
        self.GetNextToken()
        then_body = self.ParseBlock()        # Parse the body block
        self.Expect(TokenType.curlyBracClosed)
        self.GetNextToken()

        # Optionally handle 'else'
        if self.crtToken.type == TokenType.else_keyword:
            self.GetNextToken()
            self.Expect(TokenType.curlyBracOpen)
            self.GetNextToken()
            else_body = self.ParseBlock()
            self.Expect(TokenType.curlyBracClosed)
            self.GetNextToken()
            node = ASTIfElseNode(ASTConditionNode(cond_expr),
                                then_body,
                                else_body)
        else:
            node = ASTIfNode(ASTConditionNode(cond_expr),
                            then_body)

        self.stmt_type.append(node)
        return node

    # Parse a for loop, e.g. for (init; cond; incr) { ... }
    def ParseForStatement(self):
        self.Expect(TokenType.for_keyword)
        self.GetNextToken()

        self.Expect(TokenType.curvedBracketOpen  )  # '('
        self.GetNextToken()

        # Parse initialiser, which can be a declaration or assignment
        if self.crtToken.type == TokenType.let_keyword:
            init_stmt = self.ParseVarDecl()
        else:
            init_stmt = self.ParseAssignment()

        self.Expect(TokenType.semicolon)
        self.GetNextToken()

        cond_expr = self.ParseExpression()  # Condition expression
        condition = ASTConditionNode(cond_expr)

        self.Expect(TokenType.semicolon)
        self.GetNextToken()

        incr_stmt = self.ParseAssignment()  # Increment statement

        self.Expect(TokenType.curvedBracketClosed)
        self.GetNextToken()

        self.Expect(TokenType.curlyBracOpen)
        self.GetNextToken()
        for_body = self.ParseBlock()
        self.Expect(TokenType.curlyBracClosed)
        self.GetNextToken()

        node = ASTForNode(init_stmt, condition, incr_stmt, for_body)
        self.stmt_type.append(node)
        return node

    # Parse a function definition (declaration + body)
    def ParseFunction(self):
        self.GetNextToken() # consume 'fun'

        #  function name 
        self.Expect(TokenType.identifier)
        func_name = ASTVariableNode(self.crtToken.lexeme)
        self.GetNextToken()

        #  parameter list 
        self.Expect(TokenType.curvedBracketOpen)                # '('
        self.GetNextToken()
        param_list: dict[ASTVariableNode, ASTVarDeclNode] = {}

        while self.crtToken.type != TokenType.curvedBracketClosed:
            # <id> : <datatype> [ <len> ]
            self.Expect(TokenType.identifier)
            param_name = ASTVariableNode(self.crtToken.lexeme)
            self.GetNextToken()

            self.Expect(TokenType.variableDeclaration)          # ':'
            self.GetNextToken()

            self.Expect(TokenType.datatype)
            base_type = ASTDataTypeNode(self.crtToken.lexeme)
            self.GetNextToken()

            # optional “[ length ]” on a parameter
            if self.crtToken.type == TokenType.squareBracketOpen:
                self.GetNextToken()
                self.Expect(TokenType.integerType)
                length_node = ASTIntegerNode(int(self.crtToken.lexeme))
                self.GetNextToken()
                self.Expect(TokenType.squareBracketClosed)
                self.GetNextToken()
                array_node = ASTArrayNode([], length_node)
                decl = ASTVarDeclArrayNode(param_name, base_type, array_node)
                self.declared_arrays.add(param_name.lexeme)
            else:
                decl = ASTVarDeclNode(param_name, base_type, None)

            param_list[param_name] = decl

            if self.crtToken.type == TokenType.comma:
                self.GetNextToken()

        self.Expect(TokenType.curvedBracketClosed)              # ')'
        self.GetNextToken()

        #  return type 
        self.Expect(TokenType.arrow)                            # '->'
        self.GetNextToken()

        self.Expect(TokenType.datatype)
        ret_base_type = ASTDataTypeNode(self.crtToken.lexeme)
        self.GetNextToken()

        # optional “[ length ]” on the RETURN type 
        ret_len_ast = None
        if self.crtToken.type == TokenType.squareBracketOpen:
            self.GetNextToken()
            self.Expect(TokenType.integerType)
            ret_len_ast = ASTIntegerNode(int(self.crtToken.lexeme))
            self.GetNextToken()
            self.Expect(TokenType.squareBracketClosed)
            self.GetNextToken()

        # bookkeeping tables
        self.declared_functions[func_name.lexeme]         = ret_base_type.type
        self.declared_functions_params[func_name.lexeme]  = param_list

        # build the FunctionDecl AST node  
        func_decl = ASTFunctionDeclNode(func_name,ret_base_type,param_list,ret_len_ast)

        #  function body 
        self.Expect(TokenType.curlyBracOpen)                       # '{'
        self.GetNextToken()
        func_body = self.ParseBlock()
        self.Expect(TokenType.curlyBracClosed)                     # '}'
        self.GetNextToken()

        node = ASTFunctionNode(func_decl, func_body)
        self.stmt_type.append(node)
        return node

    # Parse a function call statement (and optionally a declaration via let)
    def ParseFunctionCall(self):
        have_to_decl = False
        if (self.crtToken.type == TokenType.let_keyword):
            have_to_decl = True
            self.GetNextToken()
        if have_to_decl:
            if (self.crtToken.type == TokenType.identifier):
                var_name = ASTVariableNode(self.crtToken.lexeme)
                self.GetNextToken()
            
            if (self.crtToken.type == TokenType.variableDeclaration):
                self.GetNextToken()
            
            if (self.crtToken.type == TokenType.datatype):
                var_type = ASTDataTypeNode(self.crtToken.lexeme)
                self.GetNextToken()
        
            self.declared_vars[var_name.lexeme] = var_type.type

            if (self.crtToken.type == TokenType.equals):
                self.GetNextToken()

        if self.crtToken.type == TokenType.identifier:
            if self.crtToken.lexeme in self.declared_functions.keys():
                func_name = ASTVariableNode(self.crtToken.lexeme)
            else:
                print("Semantic Error: Function not declared!")
            self.GetNextToken()
        
        if self.crtToken.type != TokenType.curvedBracketOpen :
            _raise_syntax("Character '(' does not exist!", self.crtToken)
        else:
            self.GetNextToken()
        
        args = []
        while self.crtToken.type != TokenType.curvedBracketClosed:
            arg_expr = self.ParseExpression()
            args.append(arg_expr)
            if self.crtToken.type == TokenType.comma:
                self.GetNextToken()
            elif self.crtToken.type != TokenType.curvedBracketClosed:
                _raise_syntax("Expected ',' or ')' in function call argument list", self.crtToken)

        self.GetNextToken() 

        node = ASTFunctionCallNode(func_name, args)
        self.stmt_type.append(node)
        return node

    # Look ahead a number of non-whitespace tokens and return the type
    def Peek(self, offset:int) -> TokenType:
        j = self.index
        steps = 0
        while steps < offset and j+1 < len(self.tokens):
            j += 1
            if self.tokens[j].type != TokenType.whitespace:
                steps += 1
        return self.tokens[j].type if j < len(self.tokens) else TokenType.end

    # Parse a single high-level statement (var decl, assignment, if, loop, etc)
    def ParseStatement(self):
        dispatch = {
            TokenType.if_keyword:    self.ParseIfStatement,
            TokenType.fun_keyword:   self.ParseFunction,
            TokenType.return_keyword:self.ParseReturn,
            TokenType.for_keyword:   self.ParseForStatement,
            TokenType.while_keyword: self.ParseWhileStatement,
        }

        #  Variable declarations starting with ‘let’ 
        if self.crtToken.type == TokenType.let_keyword:
            if ( self.Peek(1) == TokenType.identifier
                and self.Peek(2) == TokenType.variableDeclaration
                and self.Peek(3) == TokenType.datatype
                and self.Peek(4) == TokenType.squareBracketOpen ):
                return self.ParseVarDeclArray()
            return self.ParseVarDecl()

        #  Statements that start with an identifier 
        elif self.crtToken.type == TokenType.identifier:
            self.PredictNextToken()

            # 1. `id :` declaration (possibly an array)
            if self.nextToken.type == TokenType.variableDeclaration:
                save = self.next
                self.PredictNextToken()
                if self.nextToken.type == TokenType.squareBracketOpen:
                    self.next = save
                    return self.ParseVarDeclArray()
                self.next = save
                return self.ParseVarDecl()

            # 2. `id [`  could be array-element assignment
            if self.nextToken.type == TokenType.squareBracketOpen:
                # Find token after the closing ‘]’
                bracket_depth, look = 1, self.index + 2
                while look < len(self.tokens) and bracket_depth:
                    tok = self.tokens[look]
                    if tok.type == TokenType.squareBracketOpen:
                        bracket_depth += 1
                    elif tok.type == TokenType.squareBracketClosed:
                        bracket_depth -= 1
                    look += 1
                # If next significant token is ‘=’, treat as array assignment
                while look < len(self.tokens) and self.tokens[look].type == TokenType.whitespace:
                    look += 1
                if look < len(self.tokens) and self.tokens[look].type == TokenType.equals:
                    return self.ParseAssignmentArray()

            # 3. `id =`  → scalar assignment
            if self.nextToken.type == TokenType.equals:
                return self.ParseAssignment()

        #  PAD built-ins 
        elif self.crtToken.type in (
            TokenType.print_keyword,  TokenType.delay_keyword,
            TokenType.clear_keyword,  TokenType.write_keyword,
            TokenType.write_box_keyword):
            return self.ParsePadFunction()

        #  Structured statements 
        elif self.crtToken.type in dispatch:
            return dispatch[self.crtToken.type]()

        #  Otherwise: syntax error 
        _raise_syntax(f"Unexpected token {self.crtToken.type} "
                    f"('{self.crtToken.lexeme}')", self.crtToken)

    # Parse a block of statements, usually inside { }
    def ParseBlock(self):
        block = ASTBlockNode()
        while True:
            if self.crtToken.type in (TokenType.curlyBracClosed, TokenType.end):
                break

            stmt = self.ParseStatement()
            if stmt:
                block.add_statement(stmt)
            else:
                # Error recovery: skip tokens until we reach a safe stopping point
                while self.crtToken.type not in (TokenType.semicolon, TokenType.curlyBracClosed, TokenType.end):
                    self.GetNextToken()

            if self.crtToken.type == TokenType.semicolon:
                self.GetNextToken()

        # Add an end node at end of file
        if self.crtToken.type == TokenType.end:
            end_node = ASTEndNode()
            self.stmt_type.append(end_node)
            block.add_statement(end_node)

        return block

    # Parse an entire program and return the AST root
    def ParseProgram(self):
        self.GetNextToken()
        top_block = self.ParseBlock()

        prog = ASTProgramNode()
        for s in top_block.stmts:
            prog.add_statement(s)

        return prog       

    # Entry point: parse and populate ASTroot
    def Parse(self):        
        self.ASTroot = self.ParseProgram()



#Uncomment these to test the parser's functionality


# test_string_0 = "fun MaxInArray(x:int[8]) -> int { let m:int = 0; for (let i:int = 0; i < 8; i = i+1) { if (x[i] > m) { m = x[i]; } } return m; } let list_of_integers:int[] = [23, 54, 3, 65, 99, 120, 34, 21]; __print MaxInArray(list_of_integers);"
# test_string_1 = "let x : int = 0 as colour;"
# test_string_2 = "fun Race(p1_c:colour, p2_c:colour, score_max:int) -> int { let p1_score:int = 0; let p2_score:int = 0; while ((p1_score < score_max) and (p2_score < score_max)) { let p1_toss:int = __random_int 1000; let p2_toss:int = __random_int 1000; if (p1_toss > p2_toss) { p1_score = p1_score + 1; __write 1, p1_score, p1_c; } else { p2_score = p2_score + 1; __write 2, p2_score, p2_c; } __delay 100; } if (p2_score > p1_score) { return 2; } return 1; } let c1:colour = #ff00ff; let c2:colour = #ff0000; let m:int = __height; let w:int = Race(c1, c2, m); __print w;"
# test_string_3 = "let b:bool = not false; __print b; __print not b; "
# test_string_4 = "let b:bool = not false; __print b;"
# test_string_5 = "let x:int = 4; let y:int = -(x); __print y;"
# test_string_6 ="fun AverageOfTwo_2(x:int, y:int) -> float { return (x + y) / 2 as float;} __print AverageOfTwo_2(4, 6);"
# test_string_7 ="let count: int = 42; __print count;"
# test_string_8 ="let palette: colour[3] = [#FF0000, #00FF00, #0000FF]; let first: colour = palette[0]; __print first;"
# test_string_9 = "let flag: bool = true; if (flag) { __print 1;} else {  __print 0;}"
# test_string_10 = " for (let i: int = 1; i <= 3; i = i + 1) {for (let j: int = 1; j <= 2; j = j + 1) { __print (i * 10); }}"
# test_string_11 = "__print 3 * 2;"
# test_string_12 = "let i:int=0;while(i<3){let j:int=0;while(j<2){__print i*j;j=j+1;}i=i+1;}"
# test_string_13 = "let x:int=10;if(x>0){if(x<=10){__print x;}}else{__print 0;}"
# test_string_14 = "fun SumOfSquares(n:int) -> int { let i:int = 1; let total:int = 0; while (i <= n) { total = total + i * i; i = i + 1; } return total; } __print SumOfSquares(3);"
# test_string_15 = "fun SumArray(arr:int[5]) -> int { let i:int = 0; let total:int = 0; while (i < 5) { total = total + arr[i]; i = i + 1; } return total; } let nums:int[] = [1, 2, 3, 4, 5]; let sum_nums:int = SumArray(nums);  __print sum_nums;"
# test_string_16 = "fun IsEven(n:int) -> int { return n % 2 == 0; } __print IsEven(4);"
# test_string_17 = "__write 5, 7, #00FFCC; let c: int = __read 5, 7; __print c;"
# test_string_18 = "let nums:int [] = [5, 10, 15, 20, 25]; let idx:int   = 3; __print nums[idx];"
# test_string_19 = "fun Echo(a:int[3]) -> int {return a[0] + a[1] + a[2];}let x:int[] = [5,6,7];__print Echo(x)"
# test_string_20 = "fun MaxInArray(x:int[8]) -> int { let m:int = 0; for (let i:int = 0; i < 8; i = i+1) { if (x[i] > m) { m = x[i]; } } return m; } let list_of_integers:int[] = [23, 54, 3, 65, 99, 120, 34, 21]; let max:int = MaxInArray(list_of_integers); __print max;"
# test_string_21 = "fun draw_pattern(offset:int) -> bool { let colors:colour[] = [#FF0000, #FF7F00, #FFFF00, #00FF00, #0000FF, #4B0082, #9400D3]; for (let x:int = 0; x < __width; x = x + 3) { for (let y:int = 0; y < __height; y = y + 3) { let colorIndex:int = (x + y + offset) % 7; __write_box x, y, 2, 2, colors[colorIndex]; } } return true; } let offset:int = 0; let r:bool = false; while (true) { r = draw_pattern(offset); offset = offset + 1; __delay 10; }"
# test_string_22 = "fun color() -> colour { return (16777215 - __random_int 16777215) as colour; } fun cc(x:int, y:int, iter:int) -> bool { __print x; __print y; __print iter; while (iter > 0) { let c:colour = color(); let w:int = __random_int __width; let h:int = __random_int __height; __write w, h, c; iter = iter - 1; } return true; } let a:bool = cc(0, 0, 100000); __delay 1000;"
# test_string_23 = "fun sum_to_n(n: int) -> int { let sum: int = 0; let i: int = 1; while (i <= n) { sum = sum + i; i = i + 1; } return sum; } let limit: int = 10; let total: int = 0; for (let j: int = 1; j <= limit; j = j + 1) { if ((j % 2) == 0) { total = total + j; } } let result: int = sum_to_n(total); __print result;"
# test_string_24 = "let x: int = 5; let y: int = 10; let width: int = 15; let height: int = 8; let color: colour = #FF0000; __write_box x, y, width, height, color;"
# test_string_25 ="fun Sign(num : int) -> int { if (num < 0) { return -1; } else { if (num > 0) { return 1; } else { return 0; } } } let s1 : int = Sign(-5); let s2 : int = Sign(0); let s3 : int = Sign(6); __print s1; __print s2; __print s3;"
# test_string_26 = "let c:colour = 0 as colour; for (let i:int = 0; i < 64; i = i + 1) {c = __random_int(1677216) as colour; __clear c; __delay 16;}"
# test_string_27 = "let x : int = -7; let y : int = 2; let z : int = -y; __print x; __print z;"
# test_string_28 = "let x1 : int = 3; let y1 : int = 3; let c1 : colour = #0000ff; __write x1, y1, c1; let x2 : int = 10; let y2 : int = 10; let c2 : colour = #ff0000; __write x2, y2, c2;"
# test_string_29 = "let a: int[3] = [10, 20, 30]; __print a[1];"
# test_string_30 = "let a:int = 1; let b:int = 2; let c:int = 3; let d:int = 4; while ((a < b) and (c < d)) { a = a + 1; c = c + 1; __print a; __print c; }"
# test_string_31 = "fun color() -> colour { return (__random_int 16384257 - #f9f9f9 as int) as colour; } fun cc(x:int, y:int) -> bool { __print x; __print y; let c:colour = color(); let h:int = __random_int __height; let w:int = __random_int __width; __write w,h,c; return true; } let a:bool = cc(0, 0); __delay 1000;"
# test_string_32 = "//hello"
# test_string_33 = """let a: bool = true;
# let b: bool = false;
# __print a and b or true;         
# __print a and (b or false);        
# __print a or b and false;          """

# parser = Parser(test_string_23)
# parser.Parse()

# print_visitor = PrintNodesVisitor()
# parser.ASTroot.accept(print_visitor)

# # Semantic check
# semantic_checker = SemanticAnalyser()
# semantic_checker.check_semantics(parser.ASTroot)
# if semantic_checker.error_count == 0:
#     print("No Semantic Errors detected!")

# # Code gen
# code_gen = IRCodeGenerator()


# # walk the whole tree **once** – this adds all the IR to code_gen.code
# code_gen.generate_code(parser.ASTroot)

# # show it
# print("\nParIR Code:")
# print(code_gen.code)