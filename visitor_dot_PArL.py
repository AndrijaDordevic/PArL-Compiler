
from astnodes_PArL import *

class DotVisitor:
    """
    Walks the AST and produces a Graphviz DOT description.
    * Each AST node becomes a DOT node (n0, n1, …) with a short label.
    * Parent-child relationships become directed edges.
    """

    def __init__(self):
        self.lines = [
            'digraph AST {',
            '  rankdir = TB;',                 
            '  graph  [dpi=90,  ranksep=0.25, nodesep=0.20, margin="0.05,0.05"];',
            '  node   [shape=oval, fontsize=10, width=0.3, height=0.7];',
            '  edge   [arrowsize=0.6];'
        ]
        self._next_id  = 0                # unique id counter

    # public entry point 
    def generate(self, root) -> str:
        self._visit(root, None)
        self.lines.append("}")
        return "\n".join(self.lines)

    #  helpers 
    def _new_id(self) -> str:
        nid = f"n{self._next_id}"
        self._next_id += 1
        return nid

    def _array_len(self, node: ASTVarDeclArrayNode):
        if getattr(node, "declared_len", None) is not None:
            return node.declared_len.value

        if isinstance(node.expr, ASTArrayNode):
            return node.expr.length.value

        return "?"

    def _add_node(self, nid: str, label: str):
        label = label.replace('"', r'\"')         # escape quotes
        self.lines.append(f'    {nid} [label="{label}"];')

    def _add_edge(self, src: str, dst: str):
        self.lines.append(f"    {src} -> {dst};")

    # dispatcher 
    def _visit(self, node, parent_id: str | None):
        if node is None:
            return

        nid   = self._new_id()
        label = self._label(node)
        self._add_node(nid, label)
        if parent_id is not None:
            self._add_edge(parent_id, nid)

        #  recurse on children 
        if isinstance(node, ASTProgramNode):
            for stmt in node.stmts:
                self._visit(stmt, nid)

        elif isinstance(node, ASTBlockNode):
            for stmt in node.stmts:
                self._visit(stmt, nid)

        elif isinstance(node, ASTVarDeclNode):
            self._visit(node.id, nid)
            self._visit(node.type, nid)
            self._visit(node.expr, nid)

        elif isinstance(node, ASTVarDeclArrayNode):
            self._visit(node.id, nid)
            self._visit(node.type, nid)
            self._visit(node.expr, nid)

        elif isinstance(node, ASTArrayNode):
            for val in node.values:
                self._visit(val, nid)

        elif isinstance(node, ASTAssignmentNode):
            self._visit(node.id, nid)
            self._visit(node.expr, nid)

        elif isinstance(node, ASTAssignmentArrayNode):
            self._visit(node.id, nid)
            self._visit(node.expr, nid)

        elif isinstance(node, ASTBinaryOpNode):
            self._visit(node.left, nid)
            self._visit(node.right, nid)

        elif isinstance(node, ASTUnaryOpNode):
            self._visit(node.operand, nid)

        elif isinstance(node, ASTIfNode):
            self._visit(node.condition, nid)
            self._visit(node.if_body, nid)

        elif isinstance(node, ASTIfElseNode):
            self._visit(node.condition, nid)
            self._visit(node.if_body,  nid)
            self._visit(node.else_body, nid)

        elif isinstance(node, ASTForNode):
            self._visit(node.var_init,  nid)
            self._visit(node.condition, nid)
            self._visit(node.var_inc,   nid)
            self._visit(node.for_body,  nid)

        elif isinstance(node, ASTWhileNode):
            self._visit(node.condition,  nid)
            self._visit(node.while_body, nid)

        elif isinstance(node, ASTFunctionNode):
            self._visit(node.func_decl, nid)
            self._visit(node.func_body, nid)

        elif isinstance(node, ASTFunctionDeclNode):
            self._visit(node.func_name, nid)
            if isinstance(node.params, dict):
                for p in node.params.values():
                    self._visit(p, nid)
            self._visit(node.func_type, nid)

        elif isinstance(node, ASTFunctionCallNode):
            self._visit(node.func_name, nid)
            for arg in node.args:
                self._visit(arg, nid)

        elif isinstance(node, ASTFunctionCallExprNode):
            self._visit(node.func, nid)
            for arg in node.args:
                self._visit(arg, nid)

        elif isinstance(node, ASTArrayElementNode):
            self._visit(node.arr_node, nid)
            self._visit(node.index, nid)

        elif isinstance(node, ASTReturnNode):
            self._visit(node.val, nid)

        elif isinstance(node, ASTPrintNode):
            self._visit(node.expr, nid)

        elif isinstance(node, ASTDelayNode):
            self._visit(node.expr, nid)

        elif isinstance(node, ASTClearNode):
            self._visit(node.expr, nid)

        elif isinstance(node, ASTWriteNode):
            self._visit(node.colour, nid)
            self._visit(node.yPos,  nid)
            self._visit(node.xPos,  nid)

        elif isinstance(node, ASTWriteBoxNode):
            self._visit(node.colour, nid)
            self._visit(node.xPos,  nid)
            self._visit(node.yPos,  nid)
            self._visit(node.width, nid)
            self._visit(node.height, nid)

        elif isinstance(node, ASTReadNode):
            self._visit(node.left,  nid)
            self._visit(node.right, nid)

        elif isinstance(node, ASTModExpressionNode):
            self._visit(node.id,       nid)
            self._visit(node.operator, nid)
            self._visit(node.expr,     nid)

        elif isinstance(node, ASTReturnArrayNode):
            self._visit(node.val, nid)

        elif isinstance(node, ASTConditionNode):
            self._visit(node.expr, nid)
        
        elif isinstance(node, ASTRandomNode):
            self._visit(node.expr, nid)

        elif isinstance(node, ASTRandiNode):
            self._visit(node.expr, nid)

        elif isinstance(node, ASTCastNode):
            self._visit(node.expr, nid)

        elif isinstance(node, ASTAsExprNode):
            self._visit(node.expr, nid)

        elif isinstance(node, ASTFunctionDeclWithParamsNode):
            self._visit(node.func_name, nid)
            if isinstance(node.params, dict):
                for p in node.params.values():
                    self._visit(p, nid)
            elif isinstance(node.params, list):
                for p in node.params:
                    self._visit(p, nid)
            self._visit(node.func_type, nid)

        elif isinstance(node, ASTParenthesisNode):
            self._visit(node.expression, nid)
        
        elif isinstance(node, ASTEndNode):
            pass


    #  label helper 
    def _label(self, node) -> str:
        """Return a short, human-readable label for DOT."""
        match node:
            case ASTVariableNode():    return f"Var: {node.lexeme}"
            case ASTIntegerNode():     return f"Int: {node.value}"
            case ASTBoolNode():        return f"Bool: {node.value}"
            case ASTFloatNode():       return f"Float: {node.value}"
            case ASTColourNode():      return f"Colour"
            case ASTBinaryOpNode():    return f"BinOp: {node.binOp}"
            case ASTUnaryOpNode():     return f"Unary: {node.op}"
            case ASTOperatorNode():    return f"Op: {node.operator}"
            case ASTArrayNode():       return f"Array[{node.length.value}]"
            case ASTArrayElementNode():return "Elem"
            case ASTFunctionCallNode():return "Call"
            case ASTFunctionCallExprNode(): return "CallExpr"
            case ASTFunctionNode():    return "Function"
            case ASTBlockNode():       return "Block"
            case ASTProgramNode():     return "Program"
            case ASTIfNode():          return "If"
            case ASTIfElseNode():      return "IfElse"
            case ASTForNode():         return "For"
            case ASTWhileNode():       return "While"
            case ASTReturnNode():      return "Return"
            case ASTAssignmentNode():  return "Assign"
            case ASTAssignmentArrayNode(): return "ArrAssign"
            case ASTPrintNode():       return "Print"
            case ASTDelayNode():       return "Delay"
            case ASTClearNode():       return "Clear"
            case ASTWriteNode():       return "Write"
            case ASTWriteBoxNode():    return "WriteBox"
            case ASTReadNode():        return "Read"
            case ASTModExpressionNode():return "ModExpr"
            case ASTReturnArrayNode(): return "ReturnArr"
            case ASTVarDeclArrayNode(): return f"DeclArr[{self._array_len(node)}]"
            case ASTConditionNode():   return "Cond"
            case ASTRandomNode():      return "Random"
            case ASTRandiNode():       return "Randi"
            case ASTCastNode():        return "Cast"
            case ASTAsExprNode():      return "AsExpr"
            case ASTFunctionDeclWithParamsNode(): return "FuncDecl+Params"
            case ASTParenthesisNode(): return "()"
            case ASTWidthNode():       return f"Width:{node.value}"
            case ASTHeightNode():      return f"Height:{node.value}"
            case ASTDataTypeNode():    return f"Type:{node.type}"
            case ASTEndNode():         return "End"
            case _:                    return type(node).__name__
