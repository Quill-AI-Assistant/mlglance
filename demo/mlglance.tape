# VHS tape for the README demo GIF.
# Regenerate (after `pip install -e .` and `brew install vhs`), from the repo root:
#   vhs demo/mlglance.tape
# Records `mlglance demo --scenario overfit` long enough to show the verdict
# flip from "converging" to "overfitting" as validation turns up.

Output "assets/demo.gif"

Set Shell "bash"
Set Width 1200
Set Height 680
Set FontSize 13
Set FontFamily "Menlo"
Set Theme "Dracula"
Set Framerate 12
Set Padding 14

Hide
Type "export PS1='$ ' && clear"
Enter
Sleep 700ms
Show

Type "mlglance demo --scenario overfit"
Enter
Sleep 24s
Ctrl+C
Sleep 1200ms
