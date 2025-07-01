from astnodes_PArL import *

class PrintNodesVisitor():
    def __init__(self):
        # Name for identification
        self.name = "Print Tree Visitor"
        self.node_count = 0         # Tracks number of nodes visited (mostly for stats/debug)
        self._lines = []            # Holds the lines of pretty-printed output (for format_tree)
        self.tab_count = 0          # Indentation depth for console print version
        self._depth = 0             # Indentation depth for string/tree version
        self._counter = 0           # Counter for internal use
        self.indent = 0             # Another indentation value, if needed
        self.array_decls = {}       # Records array declarations (used for printing array elements)
        self._array_lookup = {}

    # String Tree Output (Pretty Print)

    def format_tree(self, node):
        """
        Returns a string tree of the AST (used for displaying in a UI/file).
        """
        self._lines = []
        self._visit(node)
        return "\n".join(self._lines)

    def _indent(self):
        # Returns indentation string for current depth (for pretty output)
        return '  ' * self._depth  
    
    def inc_tab_count(self):
        self.tab_count += 1

    def dec_tab_count(self):
        self.tab_count -= 1

    def _visit(self, node):
        """
        Main dynamic dispatcher: Finds and calls the appropriate visit_* method for a node.
        """
        if node is None:
            self._lines.append(f"{self._indent()}[Empty Node]")
            return
        self._counter += 1
        method = getattr(self, f'visit_{type(node).__name__}', self._generic)
        method(node)

    def _generic(self, node):
        # Called if no specific visit_* method found for the node type
        self._lines.append(f"{self._indent()}[Unknown node type: {type(node).__name__}]")

    # Node Visit Methods for Pretty Print 

    def visit_ASTDataTypeNode(self, node):
        self._lines.append(f"{self._indent()}Type node: {getattr(node, 'type', '?')}")

    def visit_ASTIntegerNode(self, node):
        self._lines.append(f"{self._indent()}Int node: {getattr(node, 'value', '?')}")

    def visit_ASTBoolNode(self, node):
        self._lines.append(f"{self._indent()}Bool node: {getattr(node, 'value', '?')}")

    def visit_ASTFloatNode(self, node):
        self._lines.append(f"{self._indent()}Float node: {getattr(node, 'value', '?')}")

    def visit_ASTColourNode(self, node):
        # Converts string hex colour to int for printing
        val = getattr(node, 'value', None)
        if isinstance(val, str) and val.startswith('#'):
            val = int(val[1:], 16)
        elif isinstance(val, str):
            val = int(val)
        self._lines.append(f"{self._indent()}Colour node: {val} (hex: #{val:06x})")

    def visit_ASTWidthNode(self, node):
        self._lines.append(f"{self._indent()}Width node: {getattr(node, 'value', '?')}")

    def visit_ASTHeightNode(self, node):
        self._lines.append(f"{self._indent()}Height node: {getattr(node, 'value', '?')}")

    def visit_ASTVariableNode(self, node):
        self._lines.append(f"{self._indent()}Variable: {getattr(node, 'lexeme', '?')}")

    def visit_ASTArrayNode(self, node):
        # Print array literal and recurse on its values
        length = getattr(node.length, 'value', '?')
        self._lines.append(f"{self._indent()}Array node of length {length}:")
        self._depth += 1
        for val in getattr(node, 'values', []):
            self._visit(val)
        self._depth -= 1

    def visit_ASTAssignmentNode(self, node):
        # Print assignment statement
        self._lines.append(f"{self._indent()}Assignment:")
        self._depth += 1
        self._visit(getattr(node, 'id', None))
        self._visit(getattr(node, 'expr', None))
        self._depth -= 1

    def visit_ASTAssignmentArrayNode(self, node):
        self._lines.append(f"{self._indent()}Array assignment:")
        self._depth += 1
        self._visit(getattr(node, 'id', None))
        self._visit(getattr(node, 'expr', None))
        self._depth -= 1

    def visit_ASTModExpressionNode(self, node):
        self._lines.append(f"{self._indent()}Modified expression:")
        self._depth += 1
        self._visit(getattr(node, 'id', None))
        self._visit(getattr(node, 'operator', None))
        self._visit(getattr(node, 'expr', None))
        self._depth -= 1

    def visit_ASTReturnNode(self, node):
        self._lines.append(f"{self._indent()}Return node:")
        self._depth += 1
        self._visit(getattr(node, 'val', None))
        self._depth -= 1

    def visit_ASTReturnArrayNode(self, node):
        self._lines.append(f"{self._indent()}Return array node:")
        self._depth += 1
        self._visit(getattr(node, 'val', None))
        self._depth -= 1

    def visit_ASTReadNode(self, node):
        self._lines.append(f"{self._indent()}Read node:")
        self._depth += 1
        self._lines.append(f"{self._indent()}X-expression:")
        self._depth += 1
        self._visit(getattr(node, 'left', None))
        self._depth -= 1
        self._lines.append(f"{self._indent()}Y-expression:")
        self._depth += 1
        self._visit(getattr(node, 'right', None))
        self._depth -= 1
        self._depth -= 1

    def visit_ASTVarDeclNode(self, node):
        self._lines.append(f"{self._indent()}Variable declaration:")
        self._depth += 1
        self._visit(getattr(node, 'id', None))
        self._visit(getattr(node, 'type', None))
        expr = getattr(node, 'expr', None)
        if expr is not None:
            self._lines.append(f"{self._indent()}Initial value:")
            self._depth += 1
            self._visit(expr)
            self._depth -= 1
        else:
            self._lines.append(f"{self._indent()}(no initialiser)")
        self._depth -= 1

    def visit_ASTVarDeclArrayNode(self, node):
        decl_len   = getattr(node, "declared_len", None)
        literal_len = node.expr.length.value if isinstance(node.expr, ASTArrayNode) else None
        length_txt = decl_len.value if decl_len is not None else literal_len
        self._lines.append(f"{self._indent()}Array variable declaration"
                        f" (length {length_txt if length_txt is not None else '?' }):")

        var_name = getattr(node.id, 'lexeme', '?')
        self._array_lookup[var_name] = node   
        self._depth += 1
        self._visit(node.id)
        self._visit(node.type)
        if node.expr is not None:
            self._visit(node.expr)
        else:
            self._lines.append(f"{self._indent()}(no array initialiser)")
        self._depth -= 1

    def visit_ASTOperatorNode(self, node):
        self._lines.append(f"{self._indent()}Operator node: {getattr(node, 'operator', '?')}")

    def visit_ASTBinaryOpNode(self, node):
        self._lines.append(f"{self._indent()}Binary operator: {getattr(node, 'binOp', '?')}")
        self._depth += 1
        self._visit(getattr(node, 'left', None))
        self._visit(getattr(node, 'right', None))
        self._depth -= 1

    def visit_ASTIfNode(self, node):
        self._lines.append(f"{self._indent()}If statement:")
        self._depth += 1
        condition = getattr(node, 'condition', None)
        self._lines.append(f"{self._indent()}Condition:")
        self._depth += 1
        if condition:
            self._visit(condition)
        else:
            self._lines.append(f"{self._indent()}Condition: {getattr(node, 'cond_full', '?')}")
        self._depth -= 1
        self._lines.append(f"{self._indent()}Body:")
        self._visit(getattr(node, 'if_body', None))
        self._depth -= 1

    def visit_ASTIfElseNode(self, node):
        self._lines.append(f"{self._indent()}If-else statement:")
        self._depth += 1
        condition = getattr(node, 'condition', None)
        self._lines.append(f"{self._indent()}Condition:")
        self._depth += 1
        if condition:
            self._visit(condition)
        else:
            self._lines.append(f"{self._indent()}Condition: {getattr(node, 'cond_full', '?')}")
        self._depth -= 1
        self._lines.append(f"{self._indent()}If-body:")
        self._visit(getattr(node, 'if_body', None))
        self._lines.append(f"{self._indent()}Else-body:")
        self._visit(getattr(node, 'else_body', None))
        self._depth -= 1

    def visit_ASTForNode(self, node):
        self._lines.append(f"{self._indent()}For loop:")
        self._depth += 1
        self._visit(getattr(node, 'var_init', None))
        condition = getattr(node, 'condition', None)
        if condition:
            self._visit(condition)
        else:
            self._lines.append(f"{self._indent()}Condition: {getattr(node, 'cond_full', '?')}")
        self._visit(getattr(node, 'var_inc', None))
        self._lines.append(f"{self._indent()}Body:")
        self._visit(getattr(node, 'for_body', None))
        self._depth -= 1

    def visit_ASTWhileNode(self, node):
        self._lines.append(f"{self._indent()}While loop:")
        self._depth += 1
        condition = getattr(node, 'condition', None)
        if condition:
            self._visit(condition)
        else:
            self._lines.append(f"{self._indent()}Condition: {getattr(node, 'cond_full', '?')}")
        self._lines.append(f"{self._indent()}Body:")
        self._visit(getattr(node, 'while_body', None))
        self._depth -= 1

    def visit_ASTFunctionCallExprNode(self, node):
        self._lines.append(f"{self._indent()}Function-call-expr:")
        self._depth += 1
        self._lines.append(f"{self._indent()}Function:")
        self._depth += 1
        self._visit(node.func)
        self._depth -= 1
        self._lines.append(f"{self._indent()}Args:")
        self._depth += 1
        for arg in node.args:
            self._visit(arg)
        self._depth -= 2          # close Args + Function-call-expr


    def visit_ASTArrayElementNode(self, node):
        self._lines.append(f"{self._indent()}Array-element:")
        self._depth += 1
        self._lines.append(f"{self._indent()}Array expression:")
        self._depth += 1
        self._visit(node.arr_node)
        self._depth -= 1
        self._lines.append(f"{self._indent()}Index:")
        self._depth += 1
        self._visit(node.index)
        self._depth -= 2          # close Index + Array-element


    def visit_ASTAsExprNode(self, node):
        self._lines.append(f"{self._indent()}As-expr (type {node.type}):")
        self._depth += 1
        self._visit(node.expr)
        self._depth -= 1


    def visit_ASTCastNode(self, node):
        tgt = getattr(node.target_type, "type", "?")
        self._lines.append(f"{self._indent()}Cast → {tgt}:")
        self._depth += 1
        self._visit(node.expr)
        self._depth -= 1


    def visit_ASTFunctionDeclNode(self, node):
        self._lines.append(f"{self._indent()}Function-decl:")
        self._depth += 1
        if node.func_name:
            self._lines.append(f"{self._indent()}Name: {node.func_name.lexeme}")
        if getattr(node, "params", None):
            self._lines.append(f"{self._indent()}Params:")
            self._depth += 1
            for p in node.params.values():
                self._visit(p)
            self._depth -= 1
        if node.func_type:
            self._lines.append(f"{self._indent()}Return type:")
            self._depth += 1
            self._visit(node.func_type)
            self._depth -= 1
        self._depth -= 1

    def visit_ASTFunctionDeclWithParamsNode(self, node):
        self.visit_ASTFunctionDeclNode(node)   # identical handling


    def visit_ASTFunctionNode(self, node):
        self._lines.append(f"{self._indent()}Function-def:")
        self._depth += 1
        self._visit(node.func_decl)
        self._lines.append(f"{self._indent()}Body:")
        self._depth += 1
        self._visit(node.func_body)
        self._depth -= 2


    def visit_ASTUnaryOpNode(self, node):
        self._lines.append(f"{self._indent()}Unary-op ({node.op}):")
        self._depth += 1
        self._visit(node.operand)
        self._depth -= 1


    def visit_ASTFunctionCallNode(self, node):
        self._lines.append(f"{self._indent()}Function-call:")
        self._depth += 1
        self._visit(node.func_name)
        if node.args:
            self._lines.append(f"{self._indent()}Args:")
            self._depth += 1
            for arg in node.args:
                self._visit(arg)
            self._depth -= 1
        self._depth -= 1


    def visit_ASTPrintNode(self, node):
        self._lines.append(f"{self._indent()}Print:")
        self._depth += 1
        self._visit(node.expr)
        self._depth -= 1


    def visit_ASTClearNode(self, node):
        self._lines.append(f"{self._indent()}Clear:")
        self._depth += 1
        self._visit(node.expr)
        self._depth -= 1


    def visit_ASTParenthesisNode(self, node):
        self._lines.append(f"{self._indent()}( … )")
        self._depth += 1
        self._visit(node.expression)
        self._depth -= 1


    def visit_ASTDelayNode(self, node):
        self._lines.append(f"{self._indent()}Delay ({node.expr.value} ms)")


    def visit_ASTWriteNode(self, node):
        self._lines.append(f"{self._indent()}Write-pixel:")
        self._depth += 1
        self._visit(node.colour)
        self._visit(node.yPos)
        self._visit(node.xPos)
        self._depth -= 1


    def visit_ASTWriteBoxNode(self, node):
        self._lines.append(f"{self._indent()}Write-box:")
        self._depth += 1
        self._visit(node.colour)
        self._visit(node.xPos)
        self._visit(node.yPos)
        self._visit(node.width)
        self._visit(node.height)
        self._depth -= 1


    def visit_ASTRandomNode(self, node):
        self._lines.append(f"{self._indent()}Random (0 … expr):")
        self._depth += 1
        self._visit(node.expr)
        self._depth -= 1


    def visit_ASTRandiNode(self, node):
        self._lines.append(f"{self._indent()}Randi:")
        self._depth += 1
        self._visit(node.expr)
        self._depth -= 1


    def visit_ASTConditionNode(self, node):
        self._lines.append(f"{self._indent()}Condition:")
        self._depth += 1
        self._visit(node.expr)
        self._depth -= 1


    def visit_ASTBlockNode(self, node):
        self._lines.append(f"{self._indent()}Block:")
        self._depth += 1
        for st in node.stmts:
            self._visit(st)
        self._depth -= 1


    def visit_ASTProgramNode(self, node):
        self._lines.append(f"{self._indent()}Program:")
        self._depth += 1
        for st in node.stmts:
            self._visit(st)
        self._depth -= 1


    def visit_ASTEndNode(self, node):
        self._lines.append(f"{self._indent()}[End]")