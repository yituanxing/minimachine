from src.minimachine.llvm_text import parse_module


def test_function_args_ignore_named_types_inside_attributes():
    module = r'''
%struct.foo = type { i64, i64 }
%struct.bar = type { i32 }

define void @f(ptr dead_on_unwind noalias writable sret(%struct.foo) align 8 %0, ptr byval(%struct.bar) align 4 %1, i32 noundef %2) {
  ret void
}
'''
    functions = parse_module(module)
    assert len(functions) == 1
    assert functions[0].args == ("%0", "%1", "%2")


def test_function_args_take_last_top_level_name_for_typed_pointer_spelling():
    module = r'''
%struct.foo = type { i64 }

define void @g(%struct.foo* %value, void (i32)* %callback, i64 %count) {
  ret void
}
'''
    functions = parse_module(module)
    assert len(functions) == 1
    assert functions[0].args == ("%value", "%callback", "%count")
