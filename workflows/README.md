OSPARC Dakota debugging workflow
==========================================

Trying to establish an easy workflow to work with OSPARC Dakota Service.

A failure during execution will make a reboot of the Dakota Service necessary, as well as of the service tied to it (as the handshake needs to be repeated). This renders the debugging process very time-consuming.

Thus, the aim of this workflow is to provide an offline version which allows easy debugging of the Dakota workflow creation using a mock model function, and seamless integration on the online OSPARC version in order to run real evaluations.

The steps in this workflow are:
- Craft your dakota.in file. You can use included python functions, and Jupyter Notebooks providing structure and rendering.
    - TODO make the functions context-dependent (eg able to tell whether in OSPARC or locally) and act accordingly, so the notebook can be uploaded directly to OSPARC and simply run without issues.
- Run using "make run". Iterate until you get the expected behaviour (you can retrieve & plot results in the notebooks). NB the mock python function will simply return an addition of the input (float) parameters. Categorical parameters are not yet integrated.
- Upload the notebook and helper files to a Jupyter Math service in OSPARC. Couple with a Dakota Service, and that with a Parallel Runner Service. Your evaluation model should be in an OSPARC template (see [here](TODO) for a full introduction and step-by-step guide on how to set up metamodeling in OSPARC), and have the same input and output keys that you had in


QUESTION: how to do for Python / OSPARC compatibility? Which kind of main.py do I need?
Probably talk / meet with Werner. Also look at Dakota Service code. I guess a function that either calls Python evaluator function, or calls Dkota Service (w same file??) would work. Think about it - now I am confused.
