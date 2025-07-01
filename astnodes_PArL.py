from typing import List

# BASE NODES 
class ASTNode:
    def __init__(self):
        self.name = "ASTNode"    

class ASTStatementNode(ASTNode):
    def __init__(self):
        self.name = "ASTStatementNode"

class ASTExpressionNode(ASTNode):
    def __init__(self):
        self.name = "ASTExpressionNode"

# PROGRAM/BLOCK NODES 
class ASTProgramNode(ASTNode):
    def __init__(self):
        self.name = "ASTProgramNode"
        self.stmts = []

    def add_statement(self, node):
        self.stmts.append(node)

    def accept(self, visitor):
        return visitor.visit_ASTProgramNode(self)

class ASTBlockNode(ASTNode):
    def __init__(self):
        self.name = "ASTBlockNode"
        self.stmts = []

    def add_statement(self, node):
        self.stmts.append(node)

    def accept(self, visitor):
        return visitor.visit_ASTBlockNode(self)  

#  EXPRESSION NODES 
class ASTVariableNode(ASTExpressionNode):
    def __init__(self, lexeme):
        self.name = "ASTVariableNode"
        self.lexeme = lexeme

    def accept(self, visitor):
        return visitor.visit_ASTVariableNode(self)

class ASTDataTypeNode(ASTExpressionNode):
    def __init__(self, t):
        self.name = "ASTDataTypeNode"
        self.type = t
    
    def accept(self, visitor):
        return visitor.visit_ASTDataTypeNode(self)

class ASTIntegerNode(ASTExpressionNode):
    def __init__(self, v):
        self.name = "ASTIntegerNode"
        self.value = v

    def accept(self, visitor):
        return visitor.visit_ASTIntegerNode(self)        

class ASTBoolNode(ASTExpressionNode):
    def __init__(self, v):
        self.name = "ASTBoolNode"
        self.value = v

    def accept(self, visitor):
        return visitor.visit_ASTBoolNode(self)  

class ASTFloatNode(ASTExpressionNode):
    def __init__(self, v):
        self.name = "ASTFloatNode"
        self.value = v

    def accept(self, visitor):
        return visitor.visit_ASTFloatNode(self)   

class ASTColourNode(ASTExpressionNode):
    def __init__(self, v):
        self.name = "ASTColourNode"
        self.value = v

    def accept(self, visitor):
        return visitor.visit_ASTColourNode(self)

class ASTWidthNode(ASTExpressionNode):
    def __init__(self, v):
        self.name = "ASTWidthNode"
        self.value = v

    def accept(self, visitor):
        return visitor.visit_ASTWidthNode(self)

class ASTHeightNode(ASTExpressionNode):
    def __init__(self, v):
        self.name = "ASTHeightNode"
        self.value = v

    def accept(self, visitor):
        return visitor.visit_ASTHeightNode(self)

class ASTConditionNode(ASTExpressionNode):
    def __init__(self, expr: ASTExpressionNode):
        super().__init__()
        self.name = "ASTConditionNode"
        self.expr = expr    
        
    def accept(self, v):
        return v.visit_ASTConditionNode(self)

class ASTReadNode(ASTExpressionNode):
    def __init__(self, left: ASTExpressionNode, right: ASTExpressionNode):
        super().__init__()
        self.name  = "ASTReadNode"
        self.left  = left
        self.right = right

    def accept(self, visitor):
        return visitor.visit_ASTReadNode(self)

class ASTBuiltinNode(ASTExpressionNode):
    def __init__(self, builtin: str, args: list[ASTExpressionNode]):
        super().__init__()
        self.name    = "ASTBuiltinNode"
        self.builtin = builtin   
        self.args    = args      

    def accept(self, visitor):
        return visitor.visit_builtin(self)
    
class ASTRandiNode(ASTExpressionNode):
    def __init__(self, expr):
        super().__init__()
        self.name = "ASTRandiNode"
        self.expr = expr  

    def accept(self, visitor):
        return visitor.visit_ASTRandiNode(self)

# ARRAY NODES 
class ASTArrayNode(ASTExpressionNode):
    def __init__(self, v, l):
        self.name = "ASTArrayNode"
        self.values = v
        self.length = l

    def accept(self, visitor):
        return visitor.visit_ASTArrayNode(self)
    
class ASTArrayElementNode(ASTExpressionNode):
    def __init__(self, index, ast_var_node):
        self.name = "ASTNewArrayElementNode"
        self.index = index
        self.arr_node = ast_var_node
    
    def accept(self, visitor):
        return visitor.visit_ASTArrayElementNode(self)

#  ASSIGNMENT & DECLARATION NODES 
class ASTAssignmentNode(ASTStatementNode):
    def __init__(self, ast_var_node, ast_expression_node):
        self.name = "ASTAssignmentNode"        
        self.id   = ast_var_node
        self.expr = ast_expression_node

    def accept(self, visitor):
        return visitor.visit_ASTAssignmentNode(self)

class ASTAssignmentArrayNode(ASTStatementNode):
    def __init__(self, ast_var_node, ast_array_node):
        self.name = "ASTAssignmentArrayNode"        
        self.id   = ast_var_node
        self.expr = ast_array_node

    def accept(self, visitor):
        return visitor.visit_ASTAssignmentArrayNode(self)

class ASTVarDeclNode(ASTStatementNode):
    def __init__(self, ast_var_node, ast_type_node, ast_expression_node):
        self.name = "ASTVarDeclNode"        
        self.id   = ast_var_node
        self.type = ast_type_node
        self.expr = ast_expression_node

    def accept(self, visitor):
        return visitor.visit_ASTVarDeclNode(self)

class ASTVarDeclArrayNode(ASTStatementNode):
    def __init__(self, ast_var_node, ast_type_node, ast_array_node):
        self.name = "ASTVarDeclArrayNode"        
        self.id   = ast_var_node
        self.type = ast_type_node
        self.expr = ast_array_node

    def accept(self, visitor):
        return visitor.visit_ASTVarDeclArrayNode(self)

#  CONTROL FLOW NODES 
class ASTIfNode:
    def __init__(self, condition, if_body):
        self.name = "ASTIfNode"
        self.condition = condition
        self.if_body = if_body

    def accept(self, visitor):
         return visitor.visit_ASTIfNode(self)

class ASTIfElseNode:
    def __init__(self, condition, if_body, else_body):
        self.name = "ASTIfElseNode"
        self.condition = condition
        self.if_body = if_body
        self.else_body = else_body

    def accept(self, visitor):
         return visitor.visit_ASTIfElseNode(self)

class ASTForNode:
    def __init__(self, initalisation, condition, incrementation, for_body):
        self.name = "ASTForNode"
        self.var_init = initalisation
        self.condition = condition
        self.var_inc = incrementation
        self.for_body = for_body

    def accept(self, visitor):
        return visitor.visit_ASTForNode(self)

class ASTWhileNode:
    def __init__(self, condition, while_body):
        self.name = "ASTWhileNode"
        self.condition = condition
        self.while_body = while_body

    def accept(self, visitor):
        return visitor.visit_ASTWhileNode(self)
    
#  FUNCTION NODES 
class ASTFunctionDeclNode(ASTNode):
    def __init__(self,ast_name, ast_type, params=None, ret_len_ast=None):            
        self.name        = "ASTFunctionDeclNode"
        self.func_name   = ast_name
        self.func_type   = ast_type
        self.params      = params or {}
        self.ret_len_ast = ret_len_ast        

    def return_size(self) -> int:
        return self.ret_len_ast.value if self.ret_len_ast else 1

    def accept(self, visitor):
        return visitor.visit_ASTFunctionDeclNode(self)

class ASTFunctionDeclWithParamsNode(ASTFunctionDeclNode):
    def __init__(self, ast_name, ast_type, params, ret_len_ast=None):          
        super().__init__(ast_name, ast_type, params, ret_len_ast)
        self.name = "ASTFunctionDeclWithParamsNode"

    def accept(self, visitor):
        return visitor.visit_ASTFunctionDeclWithParamsNode(self)

class ASTFunctionNode:
    def __init__(self, func_decl, func_body) -> None:
        self.name = "ASTFunctionNode"
        self.func_decl = func_decl
        self.func_body = func_body
    
    def accept(self, visitor):
        return visitor.visit_ASTFunctionNode(self)

class ASTFunctionCallNode:
    def __init__(self, func_name) -> None:
        self.name = "ASTFunctionCallNode"
        self.func_name = func_name
    
    def accept(self, visitor):
        return visitor.visit_ASTFunctionCallNode(self)

class ASTFunctionCallExprNode():
    def __init__(self, func: ASTVariableNode, args: List[ASTNode]):
        self.func = func
        self.args = args
    def accept(self, visitor):
        return visitor.visit_ASTFunctionCallExprNode(self)

#  OPERATOR & TYPE NODES 
class ASTParenthesisNode:
    def __init__(self, expression):
        self.expression = expression

    def accept(self, visitor):
        return visitor.visit_ASTParenthesisNode(self)

class ASTOperatorNode():
    def __init__(self, op):
        self.name = "ASTOperatorNode"
        self.operator = op

    def accept(self, visitor):
        return visitor.visit_ASTOperatorNode(self) 

class ASTBinaryOpNode():
    def __init__(self, op, left = None, right = None):
        self.name = "ASTBinaryOpNode"
        self.binOp = op
        self.left = left
        self.right = right

    def accept(self, visitor):
        return visitor.visit_ASTBinaryOpNode(self)

class ASTModExpressionNode(ASTExpressionNode):
    def __init__(self, lhs, op, rhs):
        self.name = "ASTModExpressionNode"
        self.id = lhs
        self.operator = op
        self.expr = rhs
    
    def accept(self, visitor):
        return visitor.visit_ASTModExpressionNode(self)

class ASTAsExprNode(ASTNode):
    def __init__(self, expr, target_type):
        self.expr = expr       
        self.target_type = target_type 
    def accept(self, v):
        v.visit_ASTAsExprNode(self)

class ASTUnaryOpNode(ASTExpressionNode):
    def __init__(self, op: str, operand: ASTExpressionNode):
        super().__init__()
        self.name    = "ASTUnaryOp"
        self.op      = op        
        self.operand = operand

    def accept(self, visitor):
        return visitor.visit_ASTUnaryOpNode(self)

class ASTCastNode(ASTExpressionNode):
    def __init__(self, expr: ASTExpressionNode, target_type: ASTDataTypeNode):
        super().__init__()
        self.name        = "ASTCastNode"
        self.expr        = expr
        self.target_type = target_type
    @property
    def value(self):
        return getattr(self.expr, "value", None)
    def accept(self, visitor):
        return visitor.visit_cast(self)

#  RETURN & END NODES 
class ASTReturnNode(ASTExpressionNode):
    def __init__(self, val):
        self.name = "ASTReturnNode"
        self.val = val
    
    def accept(self, visitor):
        return visitor.visit_ASTReturnNode(self)

class ASTReturnArrayNode(ASTNode):
    def __init__(self, val):
        self.val = val           
    def accept(self, v):         
        return v.visit_ASTReturnArrayNode(self)

class ASTEndNode(ASTNode):
    def __init__(self):
        self.name = "ASTEndNode"
    
    def accept(self, visitor):
        return visitor.visit_ASTEndNode(self)

#  BUILT-IN/IO/UTILITY NODES 
class ASTPrintNode:
    def __init__(self, lexeme):
        self.name = "ASTPrintNode"
        self.expr = lexeme
    
    def accept(self, visitor):
        return visitor.visit_ASTPrintNode(self)

class ASTDelayNode:
    def __init__(self, lexeme):
        self.name = "ASTDelayNode"
        self.expr = lexeme
    
    def accept(self, visitor):
        return visitor.visit_ASTDelayNode(self)

class ASTClearNode:
    def __init__(self, lexeme):
        self.name = "ASTClearNode"
        self.expr = lexeme
    
    def accept(self, visitor):
        return visitor.visit_ASTClearNode(self)

class ASTWriteNode():
    def __init__(self, colour, y, x) -> None:
        self.name = "ASTWriteNode"
        self.xPos = x       
        self.yPos = y
        self.colour = colour        
    
    def accept(self, visitor):
        return visitor.visit_ASTWriteNode(self)

class ASTWriteBoxNode():
    def __init__(self, x, y, w, h, colour) -> None:
        self.name = "ASTWriteNode"
        self.xPos = x
        self.yPos = y
        self.width = w
        self.height = h
        self.colour = colour
    
    def accept(self, visitor):
        return visitor.visit_ASTWriteBoxNode(self)

class ASTRandomNode:
    def __init__(self, lexeme):
        self.name = "ASTRandomNode"
        self.expr = lexeme
    
    def accept(self, visitor):
        return visitor.visit_ASTRandomNode(self)

#  END 
