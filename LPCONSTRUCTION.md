## Installation

Clone the repo and set up environment

```bash
git clone https://github.com/nataliepham6720/IdenfiabilitySearch

cd IdenfiabilitySearch/

pip install -r requirements.txt
pip install causaleffect
```

If one wants to plot graphs with the `plotGraph` function, either the `pycairo` library (version `1.17.2` or later) or the `cairocffi` library is also required.

## Check for identifiability

```bash 
import causaleffect

G = causaleffect.createGraph(['X->Z','Z->Y','X->Y','Z<->Y'])
causaleffect.plotGraph(G)

P = causaleffect.ID({'Y'}, {'Z'}, G, stop_on_hedge=False)
P.isUnidentifiable()  # True
P.getHedges()         # (vertices, edges, root): [((['Z', 'Y'], ['Z->Y', 'Z<->Y']), (['Y'], []))]
P.printLatex()  
```
Causal effect output: $$p(y|do(z)) = \sum_{x}P(x)P(y|do(z))$$

## Contruct LP if all criteria meet

