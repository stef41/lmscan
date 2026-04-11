# VHS tape file — https://github.com/charmbracelet/vhs
# Run: vhs demo.tape
# Produces: demo.gif

Output demo.gif

Set Shell "bash"
Set FontSize 16
Set Width 900
Set Height 500
Set Theme "Monokai"
Set Padding 20

Type "# Detect AI-generated text in one command"
Enter
Sleep 500ms

Type "pip install lmscan"
Enter
Sleep 2s

Type "lmscan 'The intricate interplay between quantum mechanics and classical physics continues to fascinate researchers across diverse disciplines, offering profound insights into the fundamental nature of reality.'"
Enter
Sleep 3s

Type "# Scan an entire directory"
Enter
Sleep 500ms

Type "lmscan --dir ./essays/"
Enter
Sleep 2s

Type "# Generate an HTML report"
Enter
Sleep 500ms

Type "lmscan --dir ./essays/ --format html"
Enter
Sleep 2s
