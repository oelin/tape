# Tape

A tiny (8 line) variable tape implementation in Python.

## Examples

Implementing a `Linear` module.

```py
from dataclasses import dataclass
from tape import tape

@dataclass
class Linear:
  features: Tuple = (2, 1)

  def __call__(self, x):
    w = tape.variable(np.random.randn(*self.features))    # weight
    b = tape.variable(np.random.randn(self.features[1]))  # bias

    return w.T @ x + b
```

Using `Linear` within a larger module (supports co-location).

```py
def model(x):
  x = Linear((28 * 28, 128))(x)
  x = Linear((128, 64))(x)
  x = Linear((64, 10))(x)

  return x
```

Run the module. Firstly initialize its parameters and then run it with those parameters.

```py
>>> batch = np.ones(28 * 28)
>>> x, variables = tape.run(model, batch, {})          # Initialization. 
>>> x, variables = tape.run(model, batch, variables)   # Forward pass.
```

Equivalent code with Flax:

```py
>>> batch = jnp.ones(28 * 28)
>>> variables = model.init(jax.random.PRNGKey(0), batch)
>>> output = model.apply(variables, batch)
```
