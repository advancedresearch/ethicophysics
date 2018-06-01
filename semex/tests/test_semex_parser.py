import vinge
from vinge.semex.parser import *

# Note: this module is called 'test_semex_parser' and not 'test_parser' because
# there is another file named 'test_parser and that creates a path conflict with
# py.test. Lame.

import pytest

class TestRegexParser:

    # Base

    def test_compile_regex_logline(self):
        r = compile_regex("logline")
        assert r == BaseAbsyn(BaseType.LOGLINE)

    def test_id(self):
        r = compile_regex("id")
        assert r == BaseAbsyn(BaseType.ID)

    def test_tag(self):
        r = compile_regex("tag")
        assert r == BaseAbsyn(BaseType.TAG)

    def test_period(self):
        r = compile_regex(".")
        assert r == BaseAbsyn(BaseType.ANYTHING)

    def test_compile_regex_parens(self):
        r = compile_regex("(logline)")
        assert r == BaseAbsyn(BaseType.LOGLINE)

    # Basic Disjunct, concat, star

    def test_star(self):
        r = compile_regex("logline*")
        assert r == StarAbsyn(BaseAbsyn(BaseType.LOGLINE))

    def test_logline_concat_id(self):
        r = compile_regex("logline id")
        assert r == ConcatAbsyn(BaseAbsyn(BaseType.LOGLINE), BaseAbsyn(BaseType.ID))

    def test_logline_or_id(self):
        r = compile_regex("logline|id")
        assert r == DisjunctAbsyn(BaseAbsyn(BaseType.LOGLINE),
                                  BaseAbsyn(BaseType.ID))

    # One level deep nesting of everything and itself

    def test_one_deep_disjunction(self):
        r = compile_regex("id|tag|logline")
        assert r == DisjunctAbsyn(BaseAbsyn(BaseType.ID),
                                  DisjunctAbsyn(BaseAbsyn(BaseType.TAG),
                                                BaseAbsyn(BaseType.LOGLINE)))

    def test_one_deep_concat(self):
        r = compile_regex("id tag logline")
        assert r == ConcatAbsyn(BaseAbsyn(BaseType.ID),
                                ConcatAbsyn(BaseAbsyn(BaseType.TAG),
                                            BaseAbsyn(BaseType.LOGLINE)))
    # More complex

    def test_logline_or_id_then_tag(self):
        r = compile_regex("logline|(id tag)")
        assert r == DisjunctAbsyn(BaseAbsyn(BaseType.LOGLINE),
                                 ConcatAbsyn(BaseAbsyn(BaseType.ID),
                                             BaseAbsyn(BaseType.TAG)))
    def test_complex_star(self):
        r = compile_regex("(id|tag)*")
        assert r == StarAbsyn(DisjunctAbsyn(BaseAbsyn(BaseType.ID),
                                            BaseAbsyn(BaseType.TAG)))

    def test_five_parens_deep(self):
        r = compile_regex("(((((logline)))))")
        assert r == BaseAbsyn(BaseType.LOGLINE)

    def test_complex_star_concat_star(self):
        r = compile_regex("(id|tag)* logline*")
        assert r == ConcatAbsyn(StarAbsyn(DisjunctAbsyn(BaseAbsyn(BaseType.ID),
                                                        BaseAbsyn(BaseType.TAG))),
                                StarAbsyn(BaseAbsyn(BaseType.LOGLINE)))

    # Failures

    def test_unbalanced_paren(self):
        with pytest.raises(RegexParseException):
            compile_regex("(")

    def test_not_logline(self):
        # 'l' then we expect 'o' for logline, but give another letter.
        # TODO(trevor) should figure out a more friendly way to propogate
        # this error up, its kind of esoteric right now.
        with pytest.raises(RegexParseException):
            compile_regex("lkgline")
