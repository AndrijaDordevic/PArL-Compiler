from astnodes_PArL import *     # AST node classes
import re                       # used for label placeholder replacement

class IRCodeGenerator():
    def __init__(self):
        self.name = "Code Generation Visitor"
        self.code = ".main\n"          # IR buffer (starts at .main)
        #  symbol / bookkeeping tables 
        self.variable_table   = {}     # scalars -> slot
        self.array_table      = {}     # arrays  -> base slot
        self.array_vals_table = {}     # not used yet, reserved
        self.array_len_table  = {}     # arrays  -> AST length node
        self.next_slot_stack  = []     # next free slot per scope
        self.function_table   = {}     # func name -> label (future use)
        self.frame_flags      = []     # does a scope open its own frame?
        self.labels           = {}     # label name -> line index
        self.scope_stack      = []     # lexical scopes (dict chain)
        self.label_count      = 0
        self.seen_program_end = False

    #  tiny helpers 
    def intialise_vars(self, var_count):
        """Emit a frame for the given number of locals (globals on .main)."""
        if var_count > 0:
            self.code += f"push {var_count}\noframe\n"
    
    def get_new_label(self):
        """Return a fresh textual label (label0, label1, …)."""
        label = f"label{self.label_count}"
        self.label_count += 1
        return label
    
    def replace_placeholders(self):
        """
        Convert '@lbl@' placeholders into PC-relative jumps once
        all label positions are known.
        """
        lines = self.code.splitlines()
        for i, line in enumerate(lines):
            for m in re.finditer(r'@([^@]+)@', line):
                lbl = m.group(1)
                if lbl not in self.labels:
                    continue
                target = self.labels[lbl]
                offset = target - i
                line = re.sub(fr'@{lbl}@', f'#PC{offset:+d}', line)
            lines[i] = line
        self.code = "\n".join(lines) + "\n"

    #  entry-point dispatcher 
    def generate_code(self, ast_root):
        """Single-dispatch front end; calls the right visitor based on type."""
        if isinstance(ast_root, ASTAssignmentNode):
            self.visit_ASTAssignmentNode(ast_root)
        if isinstance(ast_root, ASTAssignmentArrayNode):
            self.visit_ASTAssignmentArrayNode(ast_root)
        elif isinstance(ast_root, ASTVariableNode):
            self.visit_ASTVariableNode(ast_root)
        elif isinstance(ast_root, ASTConditionNode):
            self.visit_ASTConditionNode(ast_root)
        elif isinstance(ast_root, ASTVarDeclNode):
            self.visit_ASTVarDeclNode(ast_root)
        elif isinstance(ast_root, ASTVarDeclArrayNode):
            self.visit_ASTVarDeclArrayNode(ast_root)
        elif isinstance(ast_root, ASTIfNode):
            self.visit_ASTIfNode(ast_root)
        elif isinstance(ast_root, ASTCastNode):
            self.visit_ASTCastNode(ast_root)
        elif isinstance(ast_root, ASTIfElseNode):
            self.visit_ASTIfElseNode(ast_root)
        elif isinstance(ast_root, ASTForNode):
            self.visit_ASTForNode(ast_root)
        elif isinstance(ast_root, ASTWhileNode):
            self.visit_ASTWhileNode(ast_root)
        elif isinstance(ast_root, ASTBlockNode):
            self.visit_ASTBlockNode(ast_root)
        elif isinstance(ast_root, ASTProgramNode):
            self.visit_ASTProgramNode(ast_root)
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
            self.visit_ASTReadNode(ast_root)
        elif isinstance(ast_root, ASTReturnNode):
            self.visit_ASTReturnNode(ast_root)
        elif isinstance(ast_root, ASTReturnArrayNode):     
            self.visit_ASTReturnArrayNode(ast_root)
        elif isinstance(ast_root, ASTFunctionNode):
            self.visit_ASTFunctionNode(ast_root)
        elif isinstance(ast_root, ASTFunctionCallNode):
            self.visit_ASTFunctionCallNode(ast_root)
        elif isinstance(ast_root, ASTEndNode):
            self.visit_ASTEndNode(ast_root)
        return self.code

    #  scope / frame handling 
    def enter_block(self, n_locals):
        """Open a new lexical block; allocate frame if it owns locals."""
        self.scope_stack.append({})
        self.next_slot_stack.append(0) 
        self.frame_flags.append(n_locals > 0)
        if n_locals:
            self.code += f"push {n_locals}\noframe\n"
    
    def exit_block(self):
        """Close the current block; close frame if we opened one."""
        opened = self.frame_flags.pop()
        self.scope_stack.pop()
        self.next_slot_stack.pop()   

        if opened and not self.code.rstrip().endswith("ret"):
            self.code += "cframe\n"

        if not self.scope_stack:            # left the outermost block
            self.code += "halt\n"

    def count_locals_in_block(self, block):
        """Recursively count var declarations inside a block."""
        n = 0
        for stmt in block.stmts:
            if isinstance(stmt, (ASTVarDeclNode, ASTVarDeclArrayNode)):
                n += self._var_size(stmt)
            elif isinstance(stmt, ASTBlockNode):
                n += self.count_locals_in_block(stmt)
            elif hasattr(stmt, 'if_body') and hasattr(stmt, 'else_body'):
                n += self.count_locals_in_block(stmt.if_body)
                n += self.count_locals_in_block(stmt.else_body)
            elif hasattr(stmt, 'if_body'):
                n += self.count_locals_in_block(stmt.if_body)
            elif hasattr(stmt, 'while_body'):
                n += self.count_locals_in_block(stmt.while_body)
            elif hasattr(stmt, 'for_body'):
                n += self.count_locals_in_block(stmt.for_body)
        return n

    #  slot allocation / lookup 
    def _new_local(self, name, size=1):
        """Reserve <size> consecutive slots for a variable in current scope."""
        scope = self.scope_stack[-1]
        slot  = self.next_slot_stack[-1]
        scope[name] = slot
        self.next_slot_stack[-1] += size
        return slot
    
    def _var_size(self, stmt):
        if isinstance(stmt, ASTVarDeclNode):
            return 1
        if isinstance(stmt, ASTVarDeclArrayNode):
            if stmt.declared_len:              
                return stmt.declared_len.value
            if isinstance(stmt.expr, ASTArrayNode):
                return stmt.expr.length.value
            return 0
        return 0

    def _lookup(self, name):
        """Resolve <name> to (slot, depth) where depth is #frames to climb."""
        depth = 0
        for i in range(len(self.scope_stack) - 1, -1, -1):
            scope = self.scope_stack[i]
            if name in scope:
                slot = scope[name]
                return slot, depth         
            if self.frame_flags[i]:                
                depth += 1
        raise KeyError(name)
    
    def _param_name(self, p):
        """Return the identifier lexeme from various param node shapes."""
        if hasattr(p, "id"):       
            return p.id.lexeme
        else:                         
            return p.lexeme
    
    #  tiny IR emitters 
    def _emit_store(self, slot, depth):
        self.code += f"push {slot}\n"   
        self.code += f"push {depth}\n"   
        self.code += "st\n"

    def _emit_store_array(self, slot, depth, length):
        self.code += f"push {length}\n"
        self.code += f"push {slot}\n"    
        self.code += f"push {depth}\n"  
        self.code += "sta\n"

    def _push_array_values(self, name: str) -> int:
        """Push the entire array (length then data) onto the stack."""
        length = self.array_len_table[name].value
        slot, depth = self._lookup(name)
        self.code += f"push {length}\n"        
        self.code += f"pusha [{slot}:{depth}]\n"  
        return length

    #  expression dispatcher 
    def visit_expression(self, expr):
        # (order mirrors AST node classes)
        if isinstance(expr, ASTIntegerNode):
            return self.visit_ASTIntegerNode(expr)
        elif isinstance(expr, ASTBoolNode):
            return self.visit_ASTBoolNode(expr)
        elif isinstance(expr, ASTFloatNode):
            return self.visit_ASTFloatNode(expr)
        elif isinstance(expr, ASTColourNode):
            return self.visit_ASTColourNode(expr)
        elif isinstance(expr, ASTWidthNode):
            return self.visit_ASTWidthNode(expr)
        elif isinstance(expr, ASTHeightNode):
            return self.visit_ASTHeightNode(expr)
        elif isinstance(expr, ASTUnaryOpNode):
            return self.visit_ASTUnaryOpNode(expr)
        elif isinstance(expr, ASTVariableNode):
            return self.visit_ASTVariableNode(expr)
        elif isinstance(expr, ASTArrayElementNode):
            return self.visit_ASTArrayElementNode(expr)
        elif isinstance(expr, ASTBinaryOpNode):
            # evaluate right then left (stack-based)
            self.visit_expression(expr.right)
            self.visit_expression(expr.left)
            if   expr.binOp == "+": self.code += "add\n"
            elif expr.binOp == "-": self.code += "sub\n"
            elif expr.binOp == "*": self.code += "mul\n"
            elif expr.binOp == "/": self.code += "div\n"
            elif expr.binOp == "%": self.code += "mod\n"
            else:                   self.visit_ASTBinaryOpNode(expr)
        elif isinstance(expr, ASTCastNode):
            return self.visit_ASTCastNode(expr)
        elif isinstance(expr, ASTFunctionCallNode):
            return self.visit_ASTFunctionCallNode(expr)
        elif isinstance(expr, ASTFunctionCallExprNode):
            return self.visit_ASTFunctionCallExprNode(expr)
        elif isinstance(expr, ASTModExpressionNode):
            return self.visit_ASTModExpressionNode(expr)
        elif isinstance(expr, ASTRandomNode):
            return self.visit_ASTRandomNode(expr)
        elif isinstance(expr, ASTRandiNode):
            return self.visit_ASTRandiNode(expr)
        elif isinstance(expr, ASTReadNode):
            return self.visit_ASTReadNode(expr)
        else:
            raise RuntimeError(f"Unhandled expression type: {type(expr)}")

    #  leaf emitters 
    def visit_ASTIntegerNode(self, node):
        if not isinstance(node.value, (ASTWidthNode, ASTHeightNode)):
            self.code += f"push {int(node.value)}\n"

    def visit_ASTBoolNode(self, node):
        self.code += f"push {node.value}\n"

    def visit_ASTFunctionCallExprNode(self, node):
        total_args = 0                  

        for arg in reversed(node.args):
            if isinstance(arg, ASTVariableNode) and arg.lexeme in self.array_table:
                total_args += self._push_array_values(arg.lexeme)
            elif isinstance(arg, ASTArrayNode):
                total_args += 1 + len(arg.values)
                self.code += f"push {len(arg.values)}\n"
                for v in arg.values:
                    self.visit_expression(v)
            else:
                self.visit_expression(arg)
                total_args += 1

        callee = node.func.lexeme if hasattr(node, "func") \
                                   else node.func_name.lexeme
        self._emit_call(callee, total_args)

    def visit_ASTColourNode(self, node):
        val = node.value
        if isinstance(val, str) and val.startswith("#"):
            val = int(val[1:], 16)             
        self.code += f"push {val}\n"
    
    def visit_ASTWidthNode(self, node):
        self.code += "width\n"
    
    def visit_ASTHeightNode(self, node):
        self.code += "height\n"
    
    def visit_ASTArrayNode(self, node):
        for val in node.values:
            val.accept(self)

    def visit_ASTVariableNode(self, node):
        """Push variable’s current value or a literal if unresolved."""
        try:
            slot, depth = self._lookup(node.lexeme)  
            self.code += f"push [{slot}:{depth}]\n"
        except KeyError:
            self.code += f"push {node.lexeme}\n"

    def visit_ASTParanthesisNode(self, node):
        self.visit(node.expression)

    # Emit the single-token operator (+, -, *, /) after its two operands
    def visit_ASTOperatorNode(self, node):
        if (node.operator == "+"):
            self.code += "add\n"
        elif (node.operator == "-"):
            self.code += "sub\n"
        elif (node.operator == "*"):
            self.code += "mul\n"
        elif (node.operator == "/"):
            self.code += "div\n"

    # Emit comparison op after evaluating both sides (true → 1, false → 0)
    def visit_ASTBinaryOpNode(self, node):
        if   node.binOp == "<" : self.code += "lt\n"
        elif node.binOp == "<=": self.code += "le\n"
        elif node.binOp == ">" : self.code += "gt\n"
        elif node.binOp == ">=": self.code += "ge\n"
        elif node.binOp == "==": self.code += "eq\n"
        elif node.binOp == "!=": self.code += "eq\nnot\n"
        elif node.binOp == "and": self.code += "and\n"         
        elif node.binOp == "or" : self.code += "or\n"
    
    # Return statement: evaluate expr then ret
    def visit_ASTReturnNode(self, node):
            if isinstance(node.val, ASTVariableNode) \
            and node.val.lexeme in self.array_table:
                # push length then the whole array
                self._push_array_values(node.val.lexeme)
                self.code += "ret\n"
                return

            # anything else - treat as scalar
            self.visit_expression(node.val)
            self.code += "ret\n"

    def visit_ASTReturnArrayNode(self, node):
        # Literal array -- push length then elements
        if isinstance(node.val, ASTArrayNode):
            length = node.val.length.value
            self.code += f"push {length}\n"
            for v in node.val.values:
                self.visit_expression(v)

        # Named array -- push length then a pointer to its data
        elif isinstance(node.val, ASTVariableNode):
            self._push_array_values(node.val.lexeme)

        else:
            raise RuntimeError("array return must be a literal or array variable")

        self.code += "ret\n"

    # Simple scalar assignment
    def visit_ASTAssignmentNode(self, node):

        #  scalar variable on the LHS 
        if isinstance(node.id, ASTVariableNode):
            var_name = node.id.lexeme
            self.visit_expression(node.expr)
            slot, depth = self._lookup(var_name)
            self._emit_store(slot, depth)
            return                        

        #  array element on the LHS 
        if isinstance(node.id, ASTArrayElementNode):
            # 1. evaluate R-H-S  (value to be stored)
            self.visit_expression(node.expr)

            # 2. push the element length (always 1 for scalars)
            self.code += "push 1\n"

            # 3. compute element address  (base + index)
            base_slot, depth = self._lookup(node.id.arr_node.lexeme)
            self.visit_expression(node.id.index)          # index
            self.code += f"push {base_slot}\nadd\n"       # base+index

            # 4. store
            self.code += f"push {depth}\nsta\n"
            return

        # anything else on the left is a front-end bug
        raise RuntimeError("unsupported assignment shape")

    # Whole-array assignment from literal
    def visit_ASTAssignmentArrayNode(self, node):
        # whole-array assignment 
        if isinstance(node.id, ASTVariableNode):
            var_name = node.id.lexeme
            length   = node.expr.length.value             # literal’s length

            self.array_len_table[var_name] = node.expr.length
            self.code += f"push {length}\n"               # count
            for v in reversed(node.expr.values):                    # then the data
                self.visit_expression(v)

            slot, depth = self._lookup(var_name)
            self.code += f"push {slot}\n"                 # frame index
            self.code += f"push {depth}\n"                # frame level
            self.code += "sta\n"                          # store array
            return

        # single-element assignment 
        if isinstance(node.id, ASTArrayElementNode):
            self.visit_expression(node.expr)              

            self.code += "push 1\n"                       

            base_slot, depth = self._lookup(
                node.id.arr_node.lexeme)                  # constants
            self.visit_expression(node.id.index)         
            self.code += f"push {base_slot}\nadd\n"       
            self.code += f"push {depth}\n"               
            self.code += "sta\n"
            return

        raise RuntimeError("unrecognised shape for ASTAssignmentArrayNode")
    
    # Scalar var declaration with initialiser
    def visit_ASTVarDeclNode(self, node):
        slot = self._new_local(node.id.lexeme, 1)
        self.variable_table[node.id.lexeme] = slot
        self.visit_expression(node.expr)
        _, depth = self._lookup(node.id.lexeme)
        self._emit_store(slot, depth)

    # Array declaration with literal initialiser
    def visit_ASTVarDeclArrayNode(self, node):
        var_name   = node.id.lexeme
        declared_n = node.declared_len.value if node.declared_len else None

        # If the type is like int[] (no length), infer from literal
        if declared_n is None and isinstance(node.expr, ASTArrayNode):
            declared_n = len(node.expr.values)

        base_slot = self._new_local(var_name, declared_n or 0)
        self.variable_table[var_name]  = base_slot
        self.array_table[var_name]     = base_slot

        # Always set array_len_table for this array
        if declared_n is not None:
            from astnodes_PArL import ASTIntegerNode
            self.array_len_table[var_name] = ASTIntegerNode(declared_n)

        # literal initialiser 
        if isinstance(node.expr, ASTArrayNode):
            for v in reversed(node.expr.values):               
                self.visit_expression(v)
            self.code += f"push {declared_n}\n"
            _, depth = self._lookup(var_name)
            self.code += f"push {base_slot}\n"
            self.code += f"push {depth}\n"
            self.code += "sta\n"
            return                                    

        #  non-literal (function-call / another array) 
        self.visit_expression(node.expr)             
        self.code += f"push {declared_n}\n"
        _, depth = self._lookup(var_name)
        self.code += f"push {base_slot}\n"
        self.code += f"push {depth}\n"
        self.code += "sta\n"                  

    # Load a single array element (index already on stack)
    def visit_ASTArrayElementNode(self, node):
        name = node.arr_node.lexeme
        node.index.accept(self)                  # push index
        slot, depth = self._lookup(name)
        self.code += f"push +[{slot}:{depth}]\n"
    
    # “x %= y” and similar
    def visit_ASTModExpressionNode(self, node):
        self.visit_expression(node.expr)
        self.visit_expression(node.id)
        node.operator.accept(self)

    # Evaluate a condition (handles short 'and')
    def visit_ASTConditionNode(self, node):
        self.visit_expression(node.expr)

    #  control-flow visitors 
    def visit_ASTIfNode(self, node):
        self.visit_ASTConditionNode(node.condition)
        true_label = self.get_new_label()
        end_label  = self.get_new_label()
        self.code += f"push @{true_label}@\ncjmp\npush @{end_label}@\njmp\n"
        self.labels[true_label] = len(self.code.splitlines())
        self.visit_ASTBlockNode(node.if_body)
        self.labels[end_label]  = len(self.code.splitlines())

    def visit_ASTIfElseNode(self, node):
        self.visit_ASTConditionNode(node.condition)
        true_label = self.get_new_label(); else_label = self.get_new_label()
        end_label  = self.get_new_label()
        self.code += f"push @{true_label}@\ncjmp\npush @{else_label}@\njmp\n"
        self.labels[true_label] = len(self.code.splitlines())
        self.visit_ASTBlockNode(node.if_body)
        self.code += f"push @{end_label}@\njmp\n"
        self.labels[else_label] = len(self.code.splitlines())
        self.visit_ASTBlockNode(node.else_body)
        self.labels[end_label]  = len(self.code.splitlines())
    
    # Unary minus / logical not
    def visit_ASTUnaryOpNode(self, node: ASTUnaryOpNode):
        self.visit_expression(node.operand)
        if   node.op == "-":  self.code += "push 0\nsub\n"
        elif node.op == "not": self.code += "not\n"
        else: raise RuntimeError(f"Unsupported unary op {node.op}")
        
    def visit_ASTForNode(self, node):
        self.enter_block(1)                         # slot for ‘i’
        start_label = self.get_new_label()
        body_label  = self.get_new_label()
        end_label   = self.get_new_label()

        self.visit_ASTVarDeclNode(node.var_init)    # initialisation
        self.labels[start_label] = len(self.code.splitlines())
        self.visit_ASTConditionNode(node.condition) # test
        self.code += f"push @{body_label}@\ncjmp\npush @{end_label}@\njmp\n"
        self.labels[body_label] = len(self.code.splitlines())
        self.visit_ASTBlockNode(node.for_body)      # body
        # increment i
        self.code += "push 1\n"
        slot, depth = self._lookup(node.var_init.id.lexeme)
        self.code += f"push [{slot}:{depth}]\nadd\n"
        self.code += f"push {slot}\npush {depth}\nst\n"
        self.code += f"push @{start_label}@\njmp\n"
        self.labels[end_label] = len(self.code.splitlines())
        self.exit_block()
    
    # While-loop
    def visit_ASTWhileNode(self, node):
        start_label = self.get_new_label()
        body_label  = self.get_new_label()
        end_label   = self.get_new_label()

        self.labels[start_label] = len(self.code.splitlines())
        self.visit_ASTConditionNode(node.condition)
        self.code += f"push @{body_label}@\ncjmp\npush @{end_label}@\njmp\n"
        self.labels[body_label] = len(self.code.splitlines())
        self.visit_ASTBlockNode(node.while_body)
        self.code += f"push @{start_label}@\njmp\n"
        self.labels[end_label]  = len(self.code.splitlines())
    
    #  function call helpers 
    def _emit_call(self, func_label: str, n_args: int):
        self.code += f"push {n_args}\npush .{func_label}\ncall\n"

    # Function definition (slot layout + body)
    def visit_ASTFunctionNode(self, node):
        func_name = node.func_decl.func_name.lexeme
        self.code += f".{func_name}\n"

        #  build parameter frame layout 
        param_scope: dict[str, int] = {}
        next_slot = 0
        raw_params = node.func_decl.params or []
        if isinstance(raw_params, dict): raw_params = raw_params.values()

        for p in raw_params:
            pname = self._param_name(p)
            if hasattr(p, "length") or isinstance(p, ASTVarDeclArrayNode):
                arr_len_ast = p.length if hasattr(p, "length") else p.expr.length
                arr_len     = arr_len_ast.value
                param_scope[pname]          = next_slot
                self.array_table[pname]     = next_slot
                self.array_len_table[pname] = arr_len_ast
                self.variable_table[pname]  = next_slot
                next_slot += arr_len
            else:                           # scalar param
                param_scope[pname]          = next_slot
                self.variable_table[pname]  = next_slot
                next_slot += 1

        # push bookkeeping
        self.scope_stack.append(param_scope)
        self.next_slot_stack.append(next_slot)
        self.frame_flags.append(True)

        # allocate locals
        n_locals = self.count_locals_in_block(node.func_body)
        if n_locals: self.code += f"push {n_locals}\nalloc\n"

        # body (reuse frame)
        self.visit_ASTBlockNode(node.func_body, reuse_current_frame=True)

        if not self.code.rstrip().endswith("ret"):
            self.code += "push 0\nret\n"

        # pop bookkeeping
        self.scope_stack.pop(); self.next_slot_stack.pop(); self.frame_flags.pop()

    # Function call used as a statement
    def visit_ASTFunctionCallNode(self, node):
        total_args = 0                                          
        for arg in reversed(node.args):
            if isinstance(arg, ASTVariableNode) and arg.lexeme in self.array_table:
                total_args += self._push_array_values(arg.lexeme)
            elif isinstance(arg, ASTArrayNode):                
                total_args += 1 + len(arg.values)
                self.code += f"push {len(arg.values)}\n"
                for v in arg.values:
                    self.visit_expression(v)
            else:
                self.visit_expression(arg)
                total_args += 1
        self._emit_call(node.func_name.lexeme, total_args)

    #  print / IO visitors 
    def visit_ASTPrintNode(self, node):
        # If we're printing an array variable, do a manual reversed‐push + printa
        if (isinstance(node.expr, ASTVariableNode)
            and node.expr.lexeme in self.array_table):
            name   = node.expr.lexeme
            length = self.array_len_table[name].value
            slot, depth = self._lookup(name)

            # Push each element array[i] in REVERSE index order so that
            # `printa` (which pops/prints from the top of the stack) prints in forward order.
            for i in range(length - 1, -1, -1):
                # push the literal index i
                self.code += f"push {i}\n"
                # then push array[i] via “push +[slot:depth]”
                self.code += f"push +[{slot}:{depth}]\n"

            # Now push the count and call printa
            self.code += f"push {length}\n"
            self.code += "printa\n"
            return

        # Otherwise, it’s a normal scalar print
        self.visit_expression(node.expr)
        self.code += "print\n"

    def visit_ASTDelayNode(self, node):
        self.visit_expression(node.expr); self.code += "delay\n"
    
    def visit_ASTClearNode(self, node):
        self.visit_expression(node.expr); self.code += "clear\n"
    
    def visit_ASTWriteNode(self, node):
        self.visit_expression(node.colour)
        self.visit_expression(node.yPos)
        self.visit_expression(node.xPos)
        self.code += "write\n"
    
    def visit_ASTReadNode(self, node: ASTReadNode):
        self.visit_expression(node.right)
        self.visit_expression(node.left)
        self.code += "read\n"

    def visit_ASTWriteBoxNode(self, node):
        self.visit_expression(node.colour)
        self.visit_expression(node.height)
        self.visit_expression(node.width)
        self.visit_expression(node.yPos)
        self.visit_expression(node.xPos)
        self.code += "writebox\n"
    
    def visit_ASTRandomNode(self, node):
        self.visit_expression(node.expr); self.code += "irnd\n"
    
    def visit_ASTRandiNode(self, node):
        self.visit_expression(node.expr); self.code += "irandi\n"

    def visit_ASTCastNode(self, node):
        self.visit_expression(node.expr)

    #  block / program visitors 
    def visit_ASTBlockNode(self, node, reuse_current_frame=False):
        n_locals = sum(self._var_size(s) for s in node.stmts)
        if not reuse_current_frame: self.enter_block(n_locals)
        for stmt in node.stmts:     self.generate_code(stmt)
        if not reuse_current_frame: self.exit_block()

    def visit_ASTProgramNode(self, node):
        n_globals = sum(self._var_size(s) for s in node.stmts)
        self.enter_block(n_globals)

        for stmt in node.stmts:
            if isinstance(stmt, (ASTFunctionNode, ASTReturnNode)):
                continue
            self.generate_code(stmt)

        self.code += "push 0\nret\n"
        self.exit_block()

        for stmt in node.stmts:              # emit all functions
            if isinstance(stmt, ASTFunctionNode):
                self.visit_ASTFunctionNode(stmt)
        self.replace_placeholders()

    def visit_ASTEndNode(self, node):
        self.seen_program_end = True
