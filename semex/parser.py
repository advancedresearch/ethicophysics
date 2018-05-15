"""

Regex supports '|' (or), concatenation, grouping with '(' ')'
and Kleene star '*'. We do _not_ allow arbitrary repitions of kleene star.

Here is a LL(1) grammar for this regex language. This module implements
a recursive descent parser for this language, producing an abstract syntax
tree.

In the grammar below let 'eps' stand for epsilon -- the empty set.
Binding order is *, concatenation, |. Concatentation and disjunction are both
left assocatiative. There is no expicit concatenation operator, but if there
were it would be part of the factor' rule.

<regex>   ::= <exp>

<exp>     ::= <factor> <exp'>

<exp'>    ::= "|" <factor> <exp'>
            | eps

<factor>  ::= <term> <factor'>

<factor'> ::= <term> <factor'>
            | eps

<term>    ::= <base> <term'>

<term'>   ::= '*'
            | eps

<base>    ::= 'logline'
            | 'id'
            | 'tag'
            | '(' <regex> ')'
"""

class _Stream(object):
    def __init__(self, string):
        self._string = string
        self._len = len(string)
        self._idx = 0

    def peek(self):
        # Returns item or None if there's nothing left.
        if self._idx < self._len:
            return self._string[self._idx]
        return None

    def next(self):
        c = self.peek()
        self.eat(c)
        return c

    def eat(self, expected):
        c = self.peek()
        if c != expected:
            _error("expected '%s', got '%s'" % (expected, c))
        self._idx += 1

    def eat_word(self, word):
        for letter in word:
            self.eat(letter)

class RegExAbsSyn:
    """
    Base class that all abstract syntax nodes inherit from.
    There are four node classes:
      - BaseAbsyn
      - StarAbsyn
      - DisjunctAbsyn
      - ConcatAbsyn
    """
    pass

class BaseType:
    LOGLINE = 0,
    TAG = 1,
    ID = 2,
    ANYTHING = 3

class BaseAbsyn(RegExAbsSyn):
    """
    Represents leaf node.
    self.base_type is an enum, one of BaseType.
    """
    def __init__(self, base_type):
        self.base_type = base_type

    def __repr__(self):
        bt = self.base_type
        if bt == BaseType.LOGLINE:
            return 'BaseAbsyn<LogLine>'
        elif bt == BaseType.TAG:
            return 'BaseAbsyn<Tag>'
        elif bt == BaseType.ID:
            return 'BaseAbsyn<Id>'
        elif bt == BaseType.ANYTHING:
            return 'BaseAbsyn<Any>'

    def __cmp__(self, other):
        if not isinstance(other, BaseAbsyn):
            return cmp(self, other)
        return cmp(self.base_type, other.base_type)

class StarAbsyn(RegExAbsSyn):
    def __init__(self, regex):
        self.regex = regex

    def __repr__(self):
        return "%s*" % self.regex

    def __cmp__(self, other):
        if not isinstance(other, StarAbsyn):
            return cmp(self, other)
        return cmp(self.regex, other.regex)

class DisjunctAbsyn(RegExAbsSyn):
    def __init__(self, regex1, regex2):
        self.regex1 = regex1
        self.regex2 = regex2

    def __repr__(self):
        return "<%s | %s>" % (self.regex1, self.regex2)

    def __cmp__(self, other):
        if not isinstance(other, DisjunctAbsyn):
            return cmp(self, other)
        return cmp((self.regex1, self.regex2),
                   (other.regex1, other.regex2))

class ConcatAbsyn(RegExAbsSyn):
    def __init__(self, regex1, regex2):
        self.regex1 = regex1
        self.regex2 = regex2

    def __repr__(self):
        return "<%s . %s>" % (self.regex1, self.regex2)

    def __cmp__(self, other):
        if not isinstance(other, ConcatAbsyn):
            return cmp(self, other)
        return cmp((self.regex1, self.regex2),
                   (other.regex1, other.regex2))

class RegexParseException(Exception):
    """
    Exception thrown by this module when an improperly formed regex
    is encountered.
    """
    def __init__(self, message):
        self.message = message

def _error(msg):
    raise RegexParseException(msg)

def _is_id_char(c):
    return c == 'l' or c == 'i' or c == 't' or c == '.'

def _regex(stream):
    return _exp(stream)

def _exp(stream):
    c = stream.peek()
    if _is_id_char(c) or c == '(':
        f = _factor(stream)
        return _exp_prime(stream, f)
    else:
        _error("Expected <id> or '(', got '%s'" % c)

def _exp_prime(stream, token):
    c = stream.peek()
    if c == '|':
        stream.eat('|')
        factor = _factor(stream)
        token2 = _exp_prime(stream, factor)
        return DisjunctAbsyn(token, token2)
    else: # This is empty
        return token

def _factor(stream):
    c = stream.peek()
    if _is_id_char(c) or c == '(':
        term = _term(stream)
        return _factor_prime(stream, term)
    else:
        _error("Expected <id> or '(', got '%s'" % c)

def _factor_prime(stream, token):
    c = stream.peek()
    # This is concatenation
    if c == ' ':
        stream.eat(' ')
        term = _term(stream)
        token2 = _factor_prime(stream, term)
        return ConcatAbsyn(token, token2)
    else: # This is empty
        return token

def _term(stream):
    c = stream.peek()
    if _is_id_char(c) or c == '(':
        b = _base(stream)
        return _term_prime(stream, b)
    else:
        _error("Expected <id> or '(', got '%s'" % c)

def _term_prime(stream, token):
    c = stream.peek()
    if c == '*':
        stream.eat('*')
        return StarAbsyn(token)
    else:
        return token

def _base(stream):
    c = stream.peek()
    if c == 'l':
        stream.eat_word("logline")
        return BaseAbsyn(BaseType.LOGLINE)
    elif c == 'i':
        stream.eat_word("id")
        return BaseAbsyn(BaseType.ID)
    elif c == 't':
        stream.eat_word('tag')
        return BaseAbsyn(BaseType.TAG)
    elif c == '.':
        stream.eat('.')
        return BaseAbsyn(BaseType.ANYTHING)
    elif c == '(':
        stream.eat('(')
        r = _exp(stream)
        stream.eat(')')
        return r
    else:
        _error("Expected <id> or '(', got '%s'" % c)

def compile_regex(string):
    """
    Takes a regex in the form of a string and parses it into an
    abstract syntax tree. The syntax of the regex language that is parsed
    is included in this module's docstring.

    Args:
        string (str)

    Returns:
        RegExAbsSyn

    Raises:
        RegexParseException
    """
    stream = _Stream(string)
    return _regex(stream)
