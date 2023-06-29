from typing import Iterator

class Tape:

  def variable(self, x):
    if isinstance(self.tape, Iterator):
      return next(self.tape)

    return self.tape.add(x) or x

  def run(self, f, x, tape):
    self.tape = iter(tape) if tape else tape
    return f(x), tape
