# SCF-Abstract-Interface
This is a section of code in the Chronus Quantum software package that was
largely written by me to allow the SCF interface to be more general.
The design of the interface was largely designed by me with the help of 
my coworkers in the Li group.  The optimize orbitals object is largely 
composed of the former SCF code that has been refactored to utilize the
new interface.

Prior to this refactor, a different SCF function was written for each 
method. However, this generalization allowed the same algorithms to be
applied to all of our methods so far. 

Since this is only a small segment of the overall code, you can find
the rest of the code at https://urania.chem.washington.edu/chronusq/chronusq_public
