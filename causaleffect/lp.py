import itertools

from causaleffect.graph import get_directed_bidirected_graphs, get_parents, get_topological_ordering


# Define exceptions that can occur.
class MultipleUnidentifiableTerms(Exception):
    '''Exception raised when the expression contains more than one interventional term.

    Each interventional term brings its own set of response-function variables, and the
    expression multiplies them together, so the resulting program is no longer linear.'''

    def __init__(self, terms, message=None):
        self._terms = terms
        if message is None:
            clauses = ', '.join(sorted(clause_to_string(term) for term in terms))
            message = ('The expression contains ' + str(len(terms)) + ' unidentifiable terms (' +
                       clauses + '). A linear program can only be formed when at most one '
                       'term is unidentifiable.')
        super().__init__(message)


class NoUnidentifiableTerm(Exception):
    '''Exception raised when a linear program is asked for an identifiable expression.'''

    def __init__(self, message='The expression is identifiable, so there is nothing to bound.'):
        super().__init__(message)


class NotSupportedProgram(Exception):
    '''Exception raised when the expression is not a linear function of the unidentifiable term.'''
    pass


# Help functions for the unidentifiable terms of an expression
def clause_to_string(term):
    '''Function that returns the string 'P(y|do(x))' of an interventional term.'''

    return ('P(' + ', '.join(sorted(term._var)).lower() + '|do(' +
            ', '.join(sorted(term._do)).lower() + '))')


def get_unidentifiable_terms(P):
    '''Function that returns the list of interventional terms P(y|do(x)) of an expression.'''

    return P.getUnidentifiableTerms()


def count_unidentifiable_terms(P):
    '''Function that returns the number of interventional terms P(y|do(x)) of an expression.'''

    return len(P.getUnidentifiableTerms())


def check_at_most_one_unidentifiable(P, raise_error=True):
    '''Function that checks that an expression has at most one unidentifiable component.

    It returns the single interventional term P(y|do(x)) of the expression, or None if the
    expression is identifiable. If more than one interventional term is present, a
    MultipleUnidentifiableTerms exception is raised, unless raise_error is set to False, in
    which case None is returned.'''

    terms = get_unidentifiable_terms(P)
    if len(terms) > 1:
        if raise_error:
            raise MultipleUnidentifiableTerms(terms)
        return None
    if len(terms) == 0:
        return None
    return terms[0]


# LaTeX notation of the response-function variables
_STYLES = {
    # Self-contained LaTeX, it needs no macro definitions.
    'mathbf': {'q': '\\mathbf{q}', 'response': lambda name: '\\mathbf{' + name.lower() + '}'},
    # Notation of the dissertation, it needs \vqb and \vb<name> to be defined. Names that are
    # not a single word are braced, so that indexing them does not give a double subscript.
    'macro': {'q': '\\vqb', 'response': lambda name: '\\vb' + name.lower() if name.isalpha()
              else '{\\vb' + name.lower() + '}'},
}


def _style(style):
    if style not in _STYLES:
        raise ValueError("Unknown style '" + str(style) + "'. Available styles: " +
                         ', '.join(sorted(_STYLES)) + '.')
    return _STYLES[style]


def _value(name):
    '''Function that returns the LaTeX symbol of a value of a variable.'''

    return name.lower()


def _hat(latex):
    '''Function that marks every distribution of a LaTeX expression as estimated from data.'''

    return latex.replace('P(', '\\hat{P}(')


# Help functions to inspect the expression around the unidentifiable term
def _is_plain(p):
    return not (p._recursive or p._fraction or len(p._sumset) != 0 or len(p._do) != 0)


def _decompose_c_factor(q, ordering=None):
    '''Function that decomposes a c-factor into its list of (variable, conditioning set) factors.'''

    if q is None or q._fraction or len(q._sumset) != 0 or len(q._do) != 0:
        return None
    if not q._recursive:
        if not _is_plain(q):
            return None
        if len(q._var) == 1:
            (variable,) = q._var
            return [(variable, set(q._cond))]
        if ordering is None or not q._var.issubset(set(ordering)):
            return None
        factors = []
        for variable in [v for v in ordering if v in q._var]:
            previous = {v for v in ordering[:ordering.index(variable)] if v in q._var}
            factors.append((variable, set(q._cond).union(previous)))
        return factors
    factors = []
    for child in q._children:
        if not _is_plain(child) or len(child._var) != 1:
            return None
        (variable,) = child._var
        factors.append((variable, set(child._cond)))
    return factors


def _outer_factors(node, term):
    '''Recursive function that returns the summation set and the identified factors that
    multiply the interventional term inside an expression, or None if the term is not there.'''

    if node is term:
        return set(), []
    if node._fraction and node._divisor is not None:
        if len(node._divisor.getUnidentifiableTerms()) != 0:
            raise NotSupportedProgram(
                'The unidentifiable term appears in the denominator of the expression, so the '
                'objective is a ratio of linear functions instead of a linear one.')
    if not node._recursive:
        return None
    for child in node._children:
        found = _outer_factors(child, term)
        if found is None:
            continue
        if node._fraction:
            raise NotSupportedProgram(
                'The unidentifiable term appears inside a fraction, so the objective is a ratio '
                'of linear functions instead of a linear one.')
        sumset, factors = found
        others = [other for other in sorted(node._children) if other is not child]
        return node._sumset.union(sumset), factors + others
    return None


class ResponseFunction:
    '''Canonical response function of a variable of the C-component of a hedge.'''

    def __init__(self, variable, args):
        self._variable = variable
        self._args = tuple(args)

    def printLatex(self, style='mathbf'):
        '''Function that returns the LaTeX symbol of the response function itself.'''

        return _style(style)['response'](self._variable)

    def printLatexValue(self, style='mathbf', args=None):
        '''Function that returns the LaTeX symbol of the value of the response function. The
        response function is indexed by the LaTeX expressions of args, which default to the
        values of its own arguments.'''

        out = self.printLatex(style)
        if args is None:
            args = [_value(arg) for arg in self._args]
        if len(args) != 0:
            out += '_{' + ', '.join(args) + '}'
        return out

    def functions(self, cardinalities):
        '''Function that returns the list of all the mappings the response function can be.
        Each mapping is a dictionary from a tuple of values of args to a value of the variable.'''

        domain = list(itertools.product(*[range(cardinalities[arg]) for arg in self._args]))
        return [dict(zip(domain, values)) for values in
                itertools.product(range(cardinalities[self._variable]), repeat=len(domain))]

    def evaluate(self, function, values):
        '''Function that evaluates a mapping of the response function on an assignment of values.'''

        return function[tuple(values[arg] for arg in self._args)]


class CanonicalProgram:
    '''Linear program that bounds the unidentifiable term of an expression.

    The decision variables q are the probabilities of the joint response functions of the
    C-component of the hedge, the constraints match the observational c-factor Q, and the
    objective is the unidentifiable effect.'''

    def __init__(self, term, expression=None, G=None):
        self._term = term
        self._expression = expression if expression is not None else term
        self._hedge = term._hedge
        self._q = term._q
        if self._q is None:
            raise NotSupportedProgram(
                'The unidentifiable term carries no observational factor, so no constraint can '
                'be formed. The term must come from ID(..., stop_on_hedge=False).')

        ordering = get_topological_ordering(G) if G is not None else None
        factors = _decompose_c_factor(self._q, ordering)
        if factors is None:
            raise NotSupportedProgram(
                'The observational factor of the unidentifiable term is not a plain product of '
                'conditional probabilities, so its canonical parametrization is not available.' +
                ('' if G is not None else ' Passing the causal diagram G may be enough to '
                 'decompose it.'))

        self._component = _sort([variable for variable, cond in factors], factors, ordering)
        self._factors = [(variable, cond) for variable in self._component
                         for name, cond in factors if name == variable]
        # A response function only takes the parents of its variable as arguments. When the graph
        # is not given, or when a parent has already been marginalized out of the c-factor, the
        # whole conditioning set of the c-factor is taken instead: the response functions are then
        # allowed to depend on non-parents as well, which relaxes the program and widens its
        # bounds, but keeps them valid.
        G_dir = get_directed_bidirected_graphs(G)[0] if G is not None else None
        self._args = {}
        for variable, cond in factors:
            args = cond
            if G_dir is not None:
                parents = get_parents(G_dir, variable)
                if parents.issubset(cond):
                    args = parents
            self._args[variable] = tuple(_sort(args, factors, ordering))
        context = set()
        for variable, cond in factors:
            context = context.union(cond)
        self._context = _sort(context.difference(self._component), factors, ordering)
        self._index = self._context + [v for v in self._component if v not in self._context]
        self._responses = {variable: ResponseFunction(variable, self._args[variable])
                           for variable in self._component}
        self._target = _sort(term._var, factors, ordering)
        self._intervention = _sort(term._do, factors, ordering)
        self._objective_context = _sort(
            set().union(*[self._contextOf(variable) for variable in self._target]),
            factors, ordering)
        sumset, others = _outer_factors(self._expression, term) or (set(), [])
        self._outer_sumset = sumset
        self._outer_factors = others

    # Description of the program
    def component(self):
        '''Function that returns the variables of the C-component of the hedge, in topological order.'''
        return list(self._component)

    def context(self):
        '''Function that returns the variables the c-factor conditions on, in topological order.'''
        return list(self._context)

    def hedge(self):
        '''Function that returns the hedge that witnesses the unidentifiability of the effect.'''
        return self._hedge

    def _contextOf(self, variable):
        '''Recursive function that returns the context variables the value of a variable of the
        C-component depends on, once the intervention has been imposed.'''

        if variable in self._term._do:
            return set()
        out = set()
        for arg in self._args[variable]:
            if arg in self._component:
                out = out.union(self._contextOf(arg))
            else:
                out.add(arg)
        return out

    def objectiveContext(self):
        '''Function that returns the context variables the objective depends on, in topological
        order. They are the ones the weights of the objective are indexed by.'''
        return list(self._objective_context)

    def indexVariables(self):
        '''Function that returns the variables the constraints are indexed by, in topological order.'''
        return list(self._index)

    def responseFunctions(self):
        '''Function that returns the response function of each variable of the C-component.'''
        return dict(self._responses)

    def target(self):
        '''Function that returns the outcome variables of the unidentifiable effect.'''
        return list(self._target)

    def intervention(self):
        '''Function that returns the intervened variables of the unidentifiable effect.'''
        return list(self._intervention)

    # LaTeX output
    def _q_latex(self, style):
        return _style(style)['q'] + '_{' + ', '.join(
            self._responses[variable].printLatex(style) for variable in self._component) + '}'

    def _responses_latex(self, style):
        return ', '.join(self._responses[variable].printLatex(style) for variable in self._component)

    def _interventional_value(self, variable, style):
        '''Function that returns the LaTeX value of a variable of the C-component once the
        intervention has been imposed on the response functions.'''

        if variable in self._term._do:
            return _value(variable)
        args = [self._interventional_value(arg, style) if arg in self._component else _value(arg)
                for arg in self._args[variable]]
        return self._responses[variable].printLatexValue(style, args=args)

    def dataLatex(self, style='mathbf'):
        '''Function that returns the definition of the observational quantity Q the constraints
        match, in LaTeX syntax.'''

        out = 'Q(' + ', '.join(_value(v) for v in self._index) + ') = '
        for variable, cond in self._factors:
            out += '\\hat{P}(' + _value(variable)
            if len(cond) != 0:
                out += '|' + ', '.join(sorted(cond)).lower()
            out += ')'
        return out

    def objectiveLatex(self, style='mathbf'):
        '''Function that returns the objective of the program, in LaTeX syntax.'''

        return ('\\sum_{' + self._responses_latex(style) + '} c_{' + self._responses_latex(style) +
                '}' + self._q_latex(style))

    def coefficientLatex(self, style='mathbf'):
        '''Function that returns the definition of the coefficients of the objective, in LaTeX
        syntax. They are the identified part of the expression, weighted by the indicator that
        the response functions produce the target value under the intervention.'''

        out = 'c_{' + self._responses_latex(style) + '} = '
        if len(self._outer_sumset) != 0:
            out += '\\sum_{' + ', '.join(sorted(self._outer_sumset)).lower() + '}'
        for factor in self._outer_factors:
            # tab=1, so that a factor that is a sum of its own is parenthesized in the product.
            out += _hat(factor.printLatex(tab=1, simplify=False))
        out += '\\mathbb{1}\\left[' + ',\\ '.join(
            self._interventional_value(variable, style) + ' = ' + _value(variable)
            for variable in self._target) + '\\right]'
        return out

    def constraintsLatex(self, style='mathbf'):
        '''Function that returns the constraints of the program, in LaTeX syntax.'''

        conditions = ',\\ '.join(
            self._responses[variable].printLatexValue(style) + '=' + _value(variable)
            for variable in self._component)
        index = ', '.join(_value(variable) for variable in self._index)
        out = ('& \\!\\!\\sum_{\\substack{(' + self._responses_latex(style) + '):\\\\ ' +
               conditions + '}}\n    \\!\\! ' + self._q_latex(style) + '\n    \\;=\\; Q(' + index +
               ')\n    \\quad \\forall (' + index + '), \\\\\n')
        out += ('  & \\sum_{' + self._responses_latex(style) + '} ' + self._q_latex(style) +
                ' \\;=\\; 1, \\qquad \\forall ' + self._q_latex(style) + ' \\geq 0.')
        return out

    def printLatex(self, style='mathbf', label='eq:ev-primal'):
        '''Function that returns the whole bounding program, in LaTeX syntax.'''

        q = _style(style)['q']
        out = '\\begin{equation}'
        if label is not None:
            out += '\\label{' + label + '}'
        out += '\n\\begin{aligned}\n'
        out += ('\\min_{' + q + '} / \\max_{' + q + '}  &' + self.objectiveLatex(style) + '\\\\\n')
        out += '\\text{s.t.}\\quad\n  ' + self.constraintsLatex(style) + '\n'
        out += '\\end{aligned}\n\\end{equation}\n'
        out += 'where $' + self.dataLatex(style) + '$\n'
        out += 'and $' + self.coefficientLatex(style) + '$'
        return out

    def __str__(self):
        return self.printLatex()

    # Numerical form of the program
    def cardinalities(self, cardinalities=None, default=2):
        '''Function that completes the cardinalities of the variables of the program, using
        default for the variables that are not given.'''

        out = {variable: default for variable in self._index}
        if cardinalities is not None:
            for variable, value in cardinalities.items():
                out[variable] = value
        return out

    def decisionVariables(self, cardinalities=None):
        '''Function that returns the list of decision variables of the program. Each one is a
        dictionary that maps each variable of the C-component to one of its response functions,
        so it is one cell of the canonical partition of the unobserved confounders.'''

        cards = self.cardinalities(cardinalities)
        per_variable = [self._responses[variable].functions(cards) for variable in self._component]
        return [dict(zip(self._component, functions))
                for functions in itertools.product(*per_variable)]

    def _assignments(self, cardinalities):
        return [dict(zip(self._index, values)) for values in
                itertools.product(*[range(cardinalities[v]) for v in self._index])]

    def constraintMatrix(self, cardinalities=None):
        '''Function that returns the constraints of the program in numerical form.

        It returns the list of assignments of the index variables, one per constraint, and the
        matrix A of the constraints, where A[i][j] is 1 when the decision variable j produces
        the observations of the constraint i.'''

        cards = self.cardinalities(cardinalities)
        assignments = self._assignments(cards)
        columns = self.decisionVariables(cards)
        A = []
        for assignment in assignments:
            row = []
            for column in columns:
                consistent = all(
                    self._responses[variable].evaluate(column[variable], assignment)
                    == assignment[variable] for variable in self._component)
                row.append(1.0 if consistent else 0.0)
            A.append(row)
        return assignments, A

    def _interventional_values(self, column, context_values, intervention_values):
        '''Function that returns the values the C-component takes for a cell of the canonical
        partition, once the intervention has been imposed.'''

        values = dict(context_values)
        for variable in self._component:
            if variable in self._term._do:
                values[variable] = intervention_values[variable]
            else:
                values[variable] = self._responses[variable].evaluate(column[variable], values)
        return values

    def objectiveVector(self, target_values=None, intervention_values=None, context_weights=None,
                        cardinalities=None):
        '''Function that returns the coefficients of the objective in numerical form.

        target_values and intervention_values are the values of the outcome and of the
        intervention of the effect that is bounded, and they default to 1 for every variable.
        context_weights gives the weight of every assignment of the context variables the
        objective depends on, as a dictionary from a tuple of values, in the order of
        objectiveContext(), to a number. It is the identified part of the expression evaluated on
        the data, i.e. the factors reported by coefficientLatex, and it defaults to 1 when the
        objective depends on no context variable.'''

        cards = self.cardinalities(cardinalities)
        target_values = _complete(target_values, self._target)
        intervention_values = _complete(intervention_values, self._intervention)
        if context_weights is None:
            if len(self._objective_context) != 0:
                raise ValueError(
                    'The objective is a weighted sum over the assignments of (' +
                    ', '.join(self._objective_context) + '), so context_weights is required. Its '
                    'weights are the identified factors of the expression: ' +
                    self.coefficientLatex() + '.')
            context_weights = {(): 1.0}
        c = []
        for column in self.decisionVariables(cards):
            coefficient = 0.0
            for context in itertools.product(*[range(cards[v])
                                               for v in self._objective_context]):
                weight = _lookup(context_weights, context)
                if weight == 0:
                    continue
                values = self._interventional_values(
                    column, dict(zip(self._objective_context, context)), intervention_values)
                if all(values[variable] == target_values[variable] for variable in self._target):
                    coefficient += weight
            c.append(coefficient)
        return c

    def bounds(self, Q, target_values=None, intervention_values=None, context_weights=None,
               cardinalities=None):
        '''Function that returns the lower and upper bounds of the unidentifiable effect, by
        solving the program. Q gives the observational quantity of every constraint, as a
        dictionary from a tuple of values, in the order of indexVariables(), to a number.

        It requires the scipy library.'''

        try:
            from scipy.optimize import linprog
        except ImportError:
            raise ImportError('Solving the bounding program requires the scipy library. '
                              'Use constraintMatrix and objectiveVector to build the program '
                              'and solve it with another solver.')
        cards = self.cardinalities(cardinalities)
        assignments, A = self.constraintMatrix(cards)
        c = self.objectiveVector(target_values, intervention_values, context_weights, cards)
        b = [_lookup(Q, tuple(assignment[v] for v in self._index)) for assignment in assignments]
        for context in itertools.product(*[range(cards[v]) for v in self._context]):
            values = dict(zip(self._context, context))
            total = sum(value for assignment, value in zip(assignments, b)
                        if all(assignment[v] == values[v] for v in self._context))
            if abs(total - 1.0) > 1e-6:
                raise ValueError(
                    'Q is a c-factor, so it has to add up to 1 over (' +
                    ', '.join(self._component) + ') for every assignment of (' +
                    ', '.join(self._context) + '), but it adds up to ' + str(total) + ' for ' +
                    str(values) + '.')
        A_eq = A + [[1.0] * len(c)]
        b_eq = b + [1.0]
        lower = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=(0, None))
        upper = linprog([-value for value in c], A_eq=A_eq, b_eq=b_eq, bounds=(0, None))
        if not lower.success or not upper.success:
            raise ValueError('The program is infeasible: the given Q is not compatible with the '
                             'constraints. ' + str(lower.message))
        return lower.fun, -upper.fun


def _sort(variables, factors, ordering):
    '''Function that sorts variables topologically. If no ordering of the graph is given, the
    ordering is recovered from the c-factor, where every variable is conditioned on the
    variables that precede it.'''

    if ordering is not None:
        known = [v for v in ordering if v in variables]
        return known + sorted(set(variables).difference(known))
    sizes = {variable: len(cond) for variable, cond in factors}
    return sorted(variables, key=lambda v: (sizes.get(v, -1), v))


def _complete(values, variables, default=1):
    out = {variable: default for variable in variables}
    if values is not None:
        for variable, value in values.items():
            out[variable] = value
    return out


def _lookup(table, key):
    if callable(table):
        return table(*key)
    if len(key) == 1 and key not in table and key[0] in table:
        return table[key[0]]
    return table[key]


def make_program(P, G=None):
    '''Function that builds the program that bounds the unidentifiable effect of an expression.

    P is the expression returned by ID(..., stop_on_hedge=False) and G is the causal diagram,
    which gives the ordering of the variables and the arguments of the response functions. G is
    optional, but without it the response functions take the whole conditioning set of the
    c-factor as arguments, which widens the bounds of the program. It raises
    MultipleUnidentifiableTerms when more than one term of P is unidentifiable, and
    NoUnidentifiableTerm when P is identifiable.'''

    expression = P.copy()
    expression.simplify()
    if expression._recursive:
        for child in expression._children:
            child.simplify()
    term = check_at_most_one_unidentifiable(expression)
    if term is None:
        raise NoUnidentifiableTerm()
    return CanonicalProgram(term, expression=expression, G=G)