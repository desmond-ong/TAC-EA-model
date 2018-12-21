Steps to install Opensmile 2.3.0 on Mac

https://www.audeering.com/technology/opensmile/
https://www.audeering.com/download/1318/

1) Install XCode.
ACCEPT XCode's license agreement

opensmile -- install xcode, accept license agreement

2) Install Homebrew (if you haven't already)

3) Use Homebrew to install automake and libtool
`brew install automake`
`brew install libtool`


4) Follow instructions
http://www.audeering.com/research-and-open-source/files/openSMILE-book-latest.pdf

5) 
`sh buildWithPortAudio.sh`


A possible error that you might encounter 
`ld: library not found for -lopensmile`
The issue was raised here:
https://github.com/naxingyu/opensmile/issues/15
(Note that this repo owner is not the original developer of opensmile, it's just a github "mirror")

6) Test if works:
`./SMILExtract`