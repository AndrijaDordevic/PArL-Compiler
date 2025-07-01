from astnodes_PArL import *
import visitor_print_PArL

class SemanticAnalyser():
    # Set of allowed type casts for variables/expressions
    _ALLOWED_CASTS = {
        ("int",   "float"), ("float", "int"),
        ("int",   "bool"),  ("bool",  "int"),
        ("float", "bool"),  ("bool",  "float"),
        ("int",   "colour"),("colour","int")
    }
    
    def __init__(self):
        # Basic visitor and state setup
        self.name = "Semantic Analysis Visitor"
        self.scopes = [{}]                 # List of dicts, representing the stack of variable scopes
        self.error_count = 0               # Track semantic errors found
        self.array_table = {}              # Map of declared arrays
        self.function_table = {}           # Map of declared functions
        self.node_count = 0                # Can be used for stats/debug
        self.tab_count = 0                 # For pretty-printing/debug
        self.functions = {}                # Function signatures: name -> (return_type, [param_types])
        self.current_function_ret_type = None # "int", "colour", …
        self.current_function_ret_size = None # 1 ⇒ scalar, >1 ⇒ fixed-len array

    #  Entry point: main semantic check dispatcher 
    def check_semantics(self, ast_root):
        """
        Main method. Dispatches semantic checks based on the AST node type.
        Returns type for expressions, or None for statements.
        """
        if isinstance(ast_root, ASTIntegerNode):
            return self.visit_ASTIntegerNode(ast_root)
        elif isinstance(ast_root, ASTBoolNode):
            return self.visit_ASTBoolNode(ast_root)
        elif isinstance(ast_root, ASTFloatNode):
            return self.visit_ASTFloatNode(ast_root)
        elif isinstance(ast_root, ASTColourNode):
            return self.visit_ASTColourNode(ast_root)
        elif isinstance(ast_root, ASTWidthNode):
            return self.visit_ASTWidthNode(ast_root)
        elif isinstance(ast_root, ASTHeightNode):
            return self.visit_ASTHeightNode(ast_root)
        elif isinstance(ast_root, ASTCastNode):
            return self.visit_ASTCastNode(ast_root)
        elif isinstance(ast_root, ASTArrayElementNode):
            return self.visit_ASTArrayElementNode(ast_root)
        elif isinstance(ast_root, ASTBlockNode):
            return self.visit_ASTBlockNode(ast_root)
        elif isinstance(ast_root, ASTProgramNode):
            return self.visit_ASTProgramNode(ast_root)
        elif isinstance(ast_root, ASTAssignmentNode):
            self.visit_ASTAssignmentNode(ast_root)
        elif isinstance(ast_root, ASTVarDeclNode):
            self.visit_ASTVarDeclNode(ast_root)
        elif isinstance(ast_root, ASTAssignmentArrayNode):
            self.visit_ASTAssignmentArrayNode(ast_root)
        elif isinstance(ast_root, ASTVariableNode):
            self.visit_ASTVariableNode(ast_root)
        elif isinstance(ast_root, ASTConditionNode):
            self.visit_ASTConditionNode(ast_root)
        elif isinstance(ast_root, ASTUnaryOpNode):
            return self.visit_ASTUnaryOpNode(ast_root)
        elif isinstance(ast_root, ASTBinaryOpNode):
            return self.visit_ASTBinaryOpNode(ast_root)
        elif isinstance(ast_root, ASTVarDeclArrayNode):
            self.visit_ASTVarDeclArrayNode(ast_root)
        elif isinstance(ast_root, ASTIfNode):
            self.visit_ASTIfNode(ast_root)
        elif isinstance(ast_root, ASTIfElseNode):
            self.visit_ASTIfElseNode(ast_root)
        elif isinstance(ast_root, ASTForNode):
            self.visit_ASTForNode(ast_root)
        elif isinstance(ast_root, ASTWhileNode):
            self.visit_ASTWhileNode(ast_root)
        elif isinstance(ast_root, ASTCastNode):
            self.visit_ASTCastNode(ast_root)
        elif isinstance(ast_root, ASTPrintNode):
            self.visit_ASTPrintNode(ast_root)
        elif isinstance(ast_root, ASTDelayNode):
            self.visit_ASTDelayNode(ast_root)
        elif isinstance(ast_root, ASTClearNode):
            self.visit_ASTClearNode(ast_root)
        elif isinstance(ast_root, ASTWriteNode):
            self.visit_ASTWriteNode(ast_root)
        elif isinstance(ast_root, ASTWriteBoxNode):
            self.visit_ASTWriteBoxNode(ast_root)
        elif isinstance(ast_root, ASTRandomNode):
            self.visit_ASTRandomNode(ast_root)
        elif isinstance(ast_root, ASTRandiNode):
            self.visit_ASTRandiNode(ast_root)
        elif isinstance(ast_root, ASTReadNode):
            return self.visit_ASTReadNode(ast_root)
        elif isinstance(ast_root, ASTReturnNode):
            self.visit_ASTReturnNode(ast_root)
        elif isinstance(ast_root, ASTReturnArrayNode):
            self.visit_ASTReturnArrayNode(ast_root)
        elif isinstance(ast_root, ASTFunctionNode):
            self.visit_ASTFunctionNode(ast_root)
        elif isinstance(ast_root, ASTFunctionCallExprNode):
            return self.visit_ASTFunctionCallExprNode(ast_root)
        elif isinstance(ast_root, ASTEndNode):
            self.visit_ASTEndNode(ast_root)

    # Utility: Returns a printable representation of a node's value or structure
    def _pretty_val(self, node):
        if hasattr(node, "value"):
            return node.value
        elif hasattr(node, "op") and hasattr(node, "operand"):
            return f"({node.op}{self._pretty_val(node.operand)})"
        elif hasattr(node, "binOp") and hasattr(node, "left") and hasattr(node, "right"):
            return f"({self._pretty_val(node.left)} {node.binOp} {self._pretty_val(node.right)})"
        elif hasattr(node, "lexeme"):
            return node.lexeme
        elif hasattr(node, "expr"):
            return self._pretty_val(node.expr)
        elif hasattr(node, "values") and isinstance(node.values, list):
            return "[" + ", ".join(self._pretty_val(v) for v in node.values) + "]"
        else:
            return type(node).__name__

    def _infer_array_length(self, node):
        if isinstance(node, ASTArrayNode):
            return node.length.value
        if isinstance(node, ASTVariableNode):
            # we stored its length in array_table at declaration time
            return self.array_table.get(node.lexeme)
        return None
    # Scope helpers
    def current_scope(self):
        return self.scopes[-1]

    def declared(self, name):
        """Checks if a variable is declared in any scope."""
        return any(name in scope for scope in reversed(self.scopes))
    
    def declare(self, name, info=None):
        """Declares a variable in the current scope."""
        self.current_scope()[name] = info

    def _can_cast(self, src: str, dst: str) -> bool:
        """Checks if a value of type 'src' can be assigned to 'dst'."""
        return src == dst or (src, dst) in self._ALLOWED_CASTS

    # Error reporting utility
    def error(self, msg, node=None):
        """Prints a semantic error, tracking line/col if possible."""
        if node is not None and hasattr(node, "line") and hasattr(node, "col"):
            print(f"Semantic Error [Line {node.line}, Col {node.col}]: {msg}")
        else:
            print(f"Semantic Error: {msg}")
        self.error_count += 1

    #  Visitors for AST node types 

    # Each visitor below checks for type consistency, declarations, etc.

    def visit_ASTIntegerNode(self, node):
        node.inferred_type = "int"
        if not isinstance(node, ASTIntegerNode):
            self.error(f"Value {node.value} not of type integer!", node)

    def visit_ASTFunctionCallExprNode(self, node):
        sig = self.functions.get(node.func.lexeme)
        if sig is None:
            self.error(f"function '{node.func.lexeme}' not declared", node)
            return "error"

        ret_type, param_types, param_sizes = sig
        if len(param_types) != len(node.args):
            self.error("bad arg count in call to " + node.func.lexeme, node)

        for expected, expected_len, actual in zip(param_types, param_sizes, node.args):
            self.check_semantics(actual)
            t = getattr(actual, "inferred_type", None)

            if expected.endswith("[]"):
                # must be an array of the same base type and exact length
                if not (t and t.endswith("[]") and t[:-2] == expected[:-2]):
                    self.error("array element‐type mismatch in call argument", actual)
                actual_len = self._infer_array_length(actual)
                if expected_len is not None and actual_len != expected_len:
                    self.error(f"array length mismatch in call: expected {expected_len}, got {actual_len}", actual)
            else:
                if t != expected:
                    self.error("type mismatch in argument", actual)

        node.inferred_type = ret_type
        return ret_type

    def visit_ASTBoolNode(self, node):
        node.inferred_type = "bool"
        if not isinstance(node, ASTBoolNode):
            self.error(f"Value {node.value} not of type boolean!", node)

    def visit_ASTFloatNode(self, node):
        node.inferred_type = "float"
        if not isinstance(node, ASTFloatNode):
            self.error(f"Value {node.value} not of type float!", node)

    def visit_ASTColourNode(self, node):
        node.inferred_type = "colour"
        if not isinstance(node, ASTColourNode):
            self.error(f"Value {node.value} not of type colour!", node)

    def visit_ASTReadNode(self, node: ASTReadNode):
        # __read expects two ints as input
        self.check_semantics(node.left)
        self.check_semantics(node.right)
        left_t  = getattr(node.left,  "inferred_type", None)
        right_t = getattr(node.right, "inferred_type", None)
        if left_t != "int" or right_t != "int":
            self.error(f"__read expects two ints but got {left_t!r} and {right_t!r}", node)
        node.inferred_type = "int"
        return "int"

    def visit_ASTUnaryOpNode(self, node: ASTUnaryOpNode):
        # Checks unary operations like - and not
        self.check_semantics(node.operand)
        operand_type = getattr(node.operand, "inferred_type", None)
        if node.op == "-":
            if operand_type not in ("int", "float"):
                self.error(f"Cannot apply unary minus to {operand_type}", node)
            node.inferred_type = operand_type
            return operand_type
        elif node.op == "not":
            if operand_type != "bool":
                self.error(f"Cannot apply 'not' to {operand_type}", node)
            node.inferred_type = "bool"
            return "bool"
        else:
            self.error(f"Unknown unary operator '{node.op}'", node)
            node.inferred_type = None
            return None

    def visit_ASTWidthNode(self, node):
        node.inferred_type = "int"
        if not isinstance(node, ASTWidthNode):
            self.error("Not width keyword!", node)

    def visit_ASTHeightNode(self, node):
        node.inferred_type = "int"
        if not isinstance(node, ASTHeightNode):
            self.error("Not height keyword!", node)

    def visit_ASTArrayNode(self, node: ASTArrayNode):
        elem_types = []
        for v in node.values:
            self.check_semantics(v)
            elem_types.append(getattr(v, "inferred_type", None))

        # empty literal ⇒ default to int[]
        elem_type = elem_types[0] if elem_types else "int"
        if any(t != elem_type for t in elem_types):
            self.error("heterogeneous array literal", node)

        node.inferred_type = elem_type + "[]"
        # length *must* be an ASTIntegerNode per parser
        self.visit_ASTIntegerNode(node.length)
        node.length_val = node.length.value

    def visit_ASTReturnArrayNode(self, node):
            if self.current_function_ret_size is None:
                self.error("return outside of a function", node)
                return
            if self.current_function_ret_size == 1:
                self.error("function is declared to return a scalar, "
                        "but array value returned", node)

            # Check expression and element type
            self.check_semantics(node.val)
            val_type = getattr(node.val, "inferred_type", "")
            if not val_type.endswith("[]"):
                self.error("return expression is not an array", node)
            else:
                elem_type = val_type[:-2]
                if elem_type != self.current_function_ret_type:
                    self.error(f"array element type mismatch in return: "
                            f"expected {self.current_function_ret_type}, "
                            f"got {elem_type}", node)

            # Optional exact-length check
            length = self._infer_array_length(node.val)
            if length is not None and length != self.current_function_ret_size:
                self.error(f"returned array length {length} "
                        f"does not match declared {self.current_function_ret_size}", node)

    def visit_ASTVariableNode(self, node: ASTVariableNode):
        # Looks up variable type from scope stack
        name = node.lexeme
        for scope in reversed(self.scopes):
            if name in scope:
                node.inferred_type = scope[name]
                return scope[name]
        self.error(f"variable {name} not declared!", node)
        node.inferred_type = None
        return None

    def visit_ASTParenthesisNode(self, node):
        self.visit(node.expression)

    def visit_ASTOperatorNode(self, node):
        if not isinstance(node, ASTOperatorNode):
            self.error(f"Operator {node} not defined!", node)

    def visit_ASTBinaryOpNode(self, node: ASTBinaryOpNode):
        #  analyse children first 
        self.check_semantics(node.left)
        self.check_semantics(node.right)
        left_t  = getattr(node.left,  "inferred_type", None)
        right_t = getattr(node.right, "inferred_type", None)

        #  arrays are illegal in any binary op 
        if (left_t or "").endswith("[]") or (right_t or "").endswith("[]"):
            self.error("array operands are not allowed in binary expressions", node)
            node.inferred_type = None
            return None

        #  divide / mod by zero 
        if node.binOp in ("/", "%"):
            if (isinstance(node.right, ASTIntegerNode) and node.right.value == 0) or \
            (isinstance(node.right, ASTFloatNode)   and node.right.value == 0.0):
                self.error("division or modulus by zero", node)

        #  arithmetic operators 
        if node.binOp in ("+", "-", "*", "/", "%"):
            if "colour" in (left_t, right_t):
                self.error("arithmetic on colour values is undefined", node)
            node.inferred_type = "float" if "float" in (left_t, right_t) else "int"

        #  comparison operators 
        elif node.binOp in ("<", "<=", ">", ">=", "==", "!="):
            if left_t != right_t:
                self.error("cannot compare values of different types", node)
            node.inferred_type = "bool"

        # logical operators 
        elif node.binOp in ("and", "or"):
            if left_t != "bool" or right_t != "bool":
                self.error(f"cannot use '{node.binOp}' on {left_t} and {right_t}", node)
            node.inferred_type = "bool"

        #  unknown operator 
        else:
            self.error(f"unknown operator '{node.binOp}'", node)
            node.inferred_type = None

        return node.inferred_type
    
    def visit_ASTReturnNode(self, node):
            if self.current_function_ret_size is None:
                self.error("return outside of a function", node)
                return
            if self.current_function_ret_size != 1:
                self.error("function is declared to return an array, "
                        "but scalar value returned", node)

            # infer the value’s type
            self.check_semantics(node.val)
            val_type = getattr(node.val, "inferred_type", None)
            if not self._can_cast(val_type, self.current_function_ret_type):
                self.error(f"return type mismatch: expected "
                        f"{self.current_function_ret_type}, got {val_type}", node)

    def visit_ASTAssignmentNode(self, node):
        # resolve the L-H-S type 
        if isinstance(node.id, ASTVariableNode):
            name = node.id.lexeme

            if not self.declared(name):
                self.error(f"variable {name} not declared", node)
                return

            # first try the current (innermost) scope
            target_type = self.current_scope().get(name)

            # otherwise look it up in an outer scope and **cache** it
            if target_type is None:
                target_type = self.visit_ASTVariableNode(node.id)
                if target_type is None:                 # unresolved – error already reported
                    return
                self.current_scope()[name] = target_type   

        elif isinstance(node.id, ASTArrayElementNode):
            target_type = self.visit_ASTArrayElementNode(node.id)

        else:
            self.error("left-hand side is not assignable", node)
            return

        # analyse the R-H-S 
        self.check_semantics(node.expr)
        rhs_type = getattr(node.expr, "inferred_type", None)

        # type compatibility 
        if not self._can_cast(rhs_type, target_type):
            self.error(f"cannot assign {rhs_type} to {target_type}", node)

    def visit_ASTAssignmentArrayNode(self, node):
        if not isinstance(node.id, ASTVariableNode):
            self.error(f"{node.id} is not a variable", node)
            return
        name = node.id.lexeme
        if name not in self.array_table:
            self.error(f"{name} not declared as an array", node)
            return

        expected_len = self.array_table[name]
        actual_len   = len(node.expr.values)
        if expected_len != actual_len:
            self.error(f"array {name} has length {expected_len} "
                    f"but assignment provides {actual_len} values", node)

        # element type check (reuse literal checker)
        dummy_decl = ASTVarDeclArrayNode(node.id, ASTDataTypeNode(
            getattr(node.expr.values[0], "inferred_type", "int")), node.expr)
        self.visit_ASTVarDeclArrayNode(dummy_decl)
        
    def visit_ASTCastNode(self, node: ASTCastNode):
        # Ensure the inner expression has been analysed
        self.check_semantics(node.expr)

        src = getattr(node.expr, "inferred_type", None)
        dst = node.target_type.type

        if src is None:
            self.error("could not infer type of cast operand", node)
            node.inferred_type = dst
            return dst

        # identical types are always OK
        if src == dst:
            node.inferred_type = dst
            return dst

        if (src, dst) not in self._ALLOWED_CASTS:
            self.error(f"cannot cast {src} → {dst}", node)

        node.inferred_type = dst
        return dst

    def visit_ASTVarDeclNode(self, node: ASTVarDeclNode):
        # Checks for redeclaration and initialisation type
        var_name = node.id.lexeme
        var_type = node.type.type
        if self.declared(var_name):
            self.error(f"variable {var_name} already defined", node)
            return

        # Declare the variable in the current scope
        self.declare(var_name, var_type)

        # Now figure out the type of the initializer expression
        expr = node.expr
        expr_inner = None
        expr_type = None

        # 1) If it’s a unary op, cast, random, array-element, function call, etc.
        if isinstance(expr, ASTUnaryOpNode):
            expr_type    = self.visit_ASTUnaryOpNode(expr)
            expr_inner   = expr

        elif isinstance(expr, ASTCastNode):
            self.visit_ASTCastNode(expr)
            expr_inner   = expr.expr
            expr_type    = expr.target_type.type

        elif isinstance(expr, ASTRandomNode):
            self.visit_ASTRandomNode(expr)
            expr_inner   = expr
            expr_type    = "int"

        elif isinstance(expr, ASTArrayElementNode):
            expr_type    = self.visit_ASTArrayElementNode(expr)
            expr_inner   = expr

        elif isinstance(expr, ASTReadNode):
            expr_type    = self.visit_ASTReadNode(expr)
            expr_inner   = expr

        elif isinstance(expr, ASTBinaryOpNode):
            expr_type    = self.visit_ASTBinaryOpNode(expr)
            expr_inner   = expr

        elif isinstance(expr, ASTFunctionCallExprNode):
            expr_type    = self.visit_ASTFunctionCallExprNode(expr)
            expr_inner   = expr

        # 2) If it’s a plain variable, look up its type
        elif isinstance(expr, ASTVariableNode):
            expr_type    = self.visit_ASTVariableNode(expr)
            expr_inner   = expr

        # 3) Fallback for literals
        else:
            expr_inner = expr
            if isinstance(expr_inner, (ASTIntegerNode, ASTWidthNode, ASTHeightNode)):
                expr_type = "int"
            elif isinstance(expr_inner, ASTFloatNode):
                expr_type = "float"
            elif isinstance(expr_inner, ASTBoolNode):
                expr_type = "bool"
            elif isinstance(expr_inner, ASTColourNode):
                expr_type = "colour"
            else:
                expr_type = None

        # 4) Check assignment compatibility
        if not self._can_cast(expr_type, var_type):
            self.error(f"cannot initialise {var_name} ({var_type}) "
                       f"with a value of type {expr_type}", node)

        # 5) Finally visit the initializer so any nested checks still happen
        if expr_inner is not None:
            self.check_semantics(expr_inner)
    
    def visit_ASTVarDeclArrayNode(self, node):
        if not isinstance(node.id, ASTVariableNode):
            self.error("array name is not a variable node", node); return
        if not isinstance(node.type, ASTDataTypeNode):
            self.error("datatype node missing in array declaration", node); return

        var_name  = node.id.lexeme
        elem_type = node.type.type

        if self.declared(var_name):
            self.error(f"variable {var_name} already defined", node)
            return

        declared_n = (
            node.declared_len.value
            if getattr(node, "declared_len", None) is not None
            else None
        )

        if isinstance(node.expr, ASTArrayNode):
            actual_n = node.expr.length.value  
            if declared_n is not None and declared_n != actual_n:
                self.error(
                    f"array {var_name} declared length {declared_n} "
                    f"but initializer has {actual_n} elements",
                    node
                )
        else:
            actual_n = None

        if declared_n is not None:
            self.array_table[var_name] = declared_n
        elif actual_n is not None:
            self.array_table[var_name] = actual_n

        self.declare(var_name, f"{elem_type}[]")

        if isinstance(node.expr, ASTArrayNode):
            for v in node.expr.values:
                self.check_semantics(v)
                vt = getattr(v, "inferred_type", None)
                if vt != elem_type:
                    self.error(
                        f"value {self._pretty_val(v)} in array {var_name} "
                        f"is type {vt}, expected {elem_type}",
                        v
                    )
        
    def visit_ASTModeExprNode(self, node):
        node.id.accept(self)
        node.operator.accept(self)
        node.expr.accept(self)
    
    def visit_ASTArrayElementNode(self, node: ASTArrayElementNode):
        # Checks for array index and variable validity
        self.check_semantics(node.index)
        idx_type = getattr(node.index, "inferred_type", None)
        if idx_type != "int":
            self.error(f"array index must be int, got {idx_type}", node)
        arr_name = node.arr_node.lexeme
        arr_type = None
        for scope in reversed(self.scopes):
            if arr_name in scope:
                arr_type = scope[arr_name]  
                break
        if arr_type is None:
            self.error(f"array {arr_name} not declared!", node)
            node.inferred_type = None
            return None
        if not arr_type.endswith("[]"):
            self.error(f"variable {arr_name} is not an array!", node)
            node.inferred_type = None
            return None
        elem_type = arr_type[:-2]
        node.inferred_type = elem_type
        return elem_type

    def visit_ASTConditionNode(self, node: ASTConditionNode):
        self.check_semantics(node.expr)
        t = getattr(node.expr, "inferred_type", None)
        if t != "bool":
            self.error(f"condition must be bool, got {t!r}", node)
        node.inferred_type = "bool"
        return "bool"

    def visit_ASTIfNode(self, node):
        self.visit_ASTConditionNode(node.condition)
        self.visit_ASTBlockNode(node.if_body)

    def visit_ASTIfElseNode(self, node):
        self.visit_ASTConditionNode(node.condition)
        self.visit_ASTBlockNode(node.if_body)
        self.visit_ASTBlockNode(node.else_body)
    
    def visit_ASTForNode(self, node):
        # Introduce a new scope for the for-loop
        self.scopes.append({})
        self.visit_ASTVarDeclNode(node.var_init)
        self.visit_ASTConditionNode(node.condition)
        self.visit_ASTAssignmentNode(node.var_inc)
        self.visit_ASTBlockNode(node.for_body)
        self.scopes.pop()
    
    def visit_ASTWhileNode(self, node):
        self.visit_ASTConditionNode(node.condition)
        self.visit_ASTBlockNode(node.while_body)
    
    def visit_ASTFunctionNode(self, node: ASTFunctionNode):
        #  build parameter signatures 
        param_types  = []
        param_sizes  = []
        for pname_node, decl in node.func_decl.params.items():
            base = decl.type.type
            if isinstance(decl, ASTVarDeclArrayNode):
                size = decl.expr.length_hint.value if hasattr(decl.expr, "length_hint") and decl.expr.length_hint else decl.expr.length.value
                param_types.append(base + "[]")
                param_sizes.append(size)
            else:
                param_types.append(base)
                param_sizes.append(None)

        #  return signature 
        ret_base = node.func_decl.func_type.type
        ret_size = node.func_decl.return_size()
        ret_sig  = f"{ret_base}[]" if ret_size > 1 else ret_base

        # save full signature: (return, param_types, param_sizes)
        self.functions[node.func_decl.func_name.lexeme] = (ret_sig, param_types, param_sizes)

        #  set current-function context 
        save_type, save_size = self.current_function_ret_type, self.current_function_ret_size
        self.current_function_ret_type  = ret_base
        self.current_function_ret_size  = ret_size

        #  create parameter scope 
        self.scopes.append({})
        for (pname_node, decl), size in zip(node.func_decl.params.items(), param_sizes):
            pname = pname_node.lexeme
            if size is None:
                self.declare(pname, decl.type.type)
            else:
                self.declare(pname, decl.type.type + "[]")
                self.array_table[pname] = size

        #  analyse body 
        self.visit_ASTBlockNode(node.func_body)
        self.scopes.pop()

        # restore outer context
        self.current_function_ret_type, self.current_function_ret_size = save_type, save_size
    
    def visit_ASTFunctionCallNode(self, node):
        # The AST stores only the callee’s name here.
        sig = self.functions.get(node.func_name.lexeme)
        if sig is None:
            self.error(f"function '{node.func_name.lexeme}' not declared!", node)
            return
        if sig is not None:
            ret_type, param_types, param_sizes = sig
            node.inferred_type = ret_type
            return ret_type

    # I/O and Builtin nodes, just check argument types
    def visit_ASTPrintNode(self, node):
        if isinstance(node.expr, ASTIntegerNode):
            self.visit_ASTIntegerNode(node.expr)
        elif isinstance(node.expr, ASTBoolNode):
            self.visit_ASTBoolNode(node.expr)
        elif isinstance(node.expr, ASTFloatNode):
            self.visit_ASTFloatNode(node.expr)
        elif isinstance(node.expr, ASTColourNode):
            self.visit_ASTColourNode(node.expr)
        elif isinstance(node.expr, ASTVariableNode):
            self.visit_ASTVariableNode(node.expr)

    def visit_ASTDelayNode(self, node):
        if isinstance(node.expr, ASTIntegerNode):
            self.visit_ASTIntegerNode(node.expr)
        elif isinstance(node.expr, ASTBoolNode):
            self.visit_ASTBoolNode(node.expr)
        elif isinstance(node.expr, ASTFloatNode):
            self.visit_ASTFloatNode(node.expr)
        elif isinstance(node.expr, ASTColourNode):
            self.visit_ASTColourNode(node.expr)
    
    def visit_ASTClearNode(self, node):
        if isinstance(node.expr, ASTVariableNode):
            self.visit_ASTVariableNode(node.expr)
        elif isinstance(node.expr, ASTColourNode):
            self.visit_ASTColourNode(node.expr)
        elif isinstance(node.expr, ASTIntegerNode):
            self.visit_ASTIntegerNode(node.expr)
        else:
            self.error("Can only clear colour variables or colour values!", node)

    def visit_ASTWriteNode(self, node):
        # Type checking for write statement
        if isinstance(node.colour, ASTColourNode):
            self.visit_ASTColourNode(node.colour)
        elif isinstance(node.colour, ASTIntegerNode):
            self.visit_ASTIntegerNode(node.colour)
        elif isinstance(node.colour, ASTVariableNode):
            self.visit_ASTVariableNode(node.colour)
        if isinstance(node.yPos, ASTIntegerNode):
            self.visit_ASTIntegerNode(node.yPos)
        elif isinstance(node.yPos, ASTVariableNode):
            self.visit_ASTVariableNode(node.yPos)
        elif isinstance(node.yPos, ASTHeightNode):
            self.visit_ASTHeightNode(node.yPos)
        if isinstance(node.xPos, ASTIntegerNode):
            self.visit_ASTIntegerNode(node.xPos)
        elif isinstance(node.xPos, ASTVariableNode):
            self.visit_ASTVariableNode(node.xPos)
        elif isinstance(node.xPos, ASTWidthNode):
            self.visit_ASTWidthNode(node.xPos)
    
    def visit_ASTWriteBoxNode(self, node):
        # Type checking for writebox statement
        if isinstance(node.colour, ASTColourNode):
            self.visit_ASTColourNode(node.colour)
        elif isinstance(node.colour, ASTIntegerNode):
            self.visit_ASTIntegerNode(node.colour)
        elif isinstance(node.colour, ASTVariableNode):
            self.visit_ASTVariableNode(node.colour)
        if isinstance(node.yPos, ASTIntegerNode):
            self.visit_ASTIntegerNode(node.yPos)
        elif isinstance(node.yPos, ASTVariableNode):
            self.visit_ASTVariableNode(node.yPos)
        elif isinstance(node.yPos, ASTHeightNode):
            self.visit_ASTHeightNode(node.yPos)
        if isinstance(node.xPos, ASTIntegerNode):
            self.visit_ASTIntegerNode(node.xPos)
        elif isinstance(node.xPos, ASTVariableNode):
            self.visit_ASTVariableNode(node.xPos)
        elif isinstance(node.xPos, ASTWidthNode):
            self.visit_ASTWidthNode(node.xPos)
        if isinstance(node.height, ASTIntegerNode):
            self.visit_ASTIntegerNode(node.height)
        elif isinstance(node.height, ASTVariableNode):
            self.visit_ASTVariableNode(node.height)
        elif isinstance(node.height, ASTHeightNode):
            self.visit_ASTHeightNode(node.height)
        if isinstance(node.width, ASTIntegerNode):
            self.visit_ASTIntegerNode(node.width)
        elif isinstance(node.width, ASTVariableNode):
            self.visit_ASTVariableNode(node.width)
        elif isinstance(node.width, ASTWidthNode):
            self.visit_ASTWidthNode(node.width)
    
    def visit_ASTRandomNode(self, node):
        node.inferred_type = "int"
        if isinstance(node.expr, ASTIntegerNode):
            self.visit_ASTIntegerNode(node.expr)
        elif isinstance(node.expr, ASTBoolNode):
            self.visit_ASTBoolNode(node.expr)
        elif isinstance(node.expr, ASTFloatNode):
            self.visit_ASTFloatNode(node.expr)
        elif isinstance(node.expr, ASTColourNode):
            self.visit_ASTColourNode(node.expr)
        elif isinstance(node.expr, ASTWidthNode):
            self.visit_ASTWidthNode(node.expr)
        elif isinstance(node.expr, ASTHeightNode):
            self.visit_ASTHeightNode(node.expr)
        else:
            self.check_semantics(node.expr)
    
    def visit_ASTRandiNode(self, node: ASTRandiNode):
        node.inferred_type = "int"
        if isinstance(node.expr, ASTIntegerNode):
            self.visit_ASTIntegerNode(node.expr)
        elif isinstance(node.expr, ASTBoolNode):
            self.visit_ASTBoolNode(node.expr)
        elif isinstance(node.expr, ASTFloatNode):
            self.visit_ASTFloatNode(node.expr)
        elif isinstance(node.expr, ASTColourNode):
            self.visit_ASTColourNode(node.expr)
        elif isinstance(node.expr, ASTWidthNode):
            self.visit_ASTWidthNode(node.expr)
        elif isinstance(node.expr, ASTHeightNode):
            self.visit_ASTHeightNode(node.expr)
        else:
            self.check_semantics(node.expr)

    # Block and program nodes: push/pop scopes
    def visit_ASTBlockNode(self, node):
        self.scopes.append({})
        for stmt in node.stmts:
            self.check_semantics(stmt)
        self.scopes.pop() 
    
    def visit_ASTProgramNode(self, node):
        for stmt in node.stmts:
            self.check_semantics(stmt)
    
    def visit_ASTEndNode(self, node):
        # Final report at program end
        print(f'Number of semantic errors: {self.error_count}')
