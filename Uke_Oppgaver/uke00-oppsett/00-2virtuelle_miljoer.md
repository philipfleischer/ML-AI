# Virtuelle miljøer i Python

I dette dokumentet skal vi se på hva virtuelle miljøer (virtual environments) er, hvorfor man bruker det og hvordan man kan bruke det i Python. Konseptet rundt virtuelle miljøer er relativt enkelt: man lager distinkte steder der man kan installere pakker hver for seg. Siden det er et viktig og mye brukt konsept kan det lønne seg å bruke litt tid på å forstå hva det faktisk er. Virtuelle miljøer er essentielle for store prosjekter og dersom dere programmerer i jobbsammenheng.

## Bakgrunn

Vi starter med å se på hvilket problem som virtual environment løser.

Når man programmerer vil man ofte bruke funksjonalitet som allerede er laget av andre. Dette kan man gjøre ved å importere pakker / biblioteker, slik som `NumPy` og `pandas` i Python. Disse pakkene vil igjen være avhengig av mange andre biblioteker, som igjen kan ha sine avhengigheter. De fleste pakker er også under utvikling og blir jevnlig oppdatert med forskjellige versjonsnummer. Noen ganger vil en ny versjon føre til at tidligere funksjonalitet ikke lenger er tilgjengelig. Dette fører til at noen pakker kan være avhengig av en pakke med en spesifikk versjon, mens andre pakker trenger et annet versjonsnummer av samme avhengighet.

Dersom man jobber på flere prosjekter vil dette raskt kunne føre til konflikter, der et prosjekt krever enkelte versjoner av biblioteker og andre prosjekter krever forskjellige versjoner. Det hadde vært upraktisk om man måtte ha reinstallert avhengigheter hver gang man byttet hvilket prosjekt man jobbet på, så derfor finnes heldigvis *virtual environments*. De vil også hjelpe med å holde oversikt over pakker og hindre at man ikke krever for mange avhengigheter på små prosjekter.

Virtuelle miljøer lager et software skille mellom hvilke pakker som er installert. Dersom man installerer noe i et virtuelt miljø vil det bare bli installert i det ene miljøet, ikke andre steder. På denne måten kan man lage ett virtuelt miljø per prosjekt man jobber på og installere akkurat de pakkene og versjonen av pakkene man trenger i hvert miljø. Disse installasjonene vil ikke påvirke andre miljøer. Når man går fra et prosjekt til et annet, kan man raskt kjøre en kommando i terminalen for å bytte miljø, som automatisk vil endre hvilke pakker som er tilgjengelige.

Dersom ingen miljøer er aktivert er dere i "root", også kalt "base". Dersom dere installerer her vil pakkene være tilgjengelig i alle miljøer. Dette er ikke anbefalt, da det kan føre til konflikter i pakkene og forvirring. Dersom dere med uhell installerer noe i root kan dere avinstallere med `pip uninstall <name-of-package>`.

## Hvordan bruke virtual environments i Python

Vi skal nå se på alt vi trenger å vite om å lage og bruke virtuelle miljøer i Python. Vi starter med å lage et nytt virtuelt miljø. Selve det virtuelle miljøet vil være en mappe som inneholder installasjonene og vil bli lagret samme sted som kommandoen for å lage miljøet som blir kjørt. Det er anbefalt å gjøre dette i roten av prosjektet. Bruk derfor `cd` i terminalen til å finne frem til der du har lagret dette repositoriet lokalt på datamaskinen. Deretter kan følgende kommando kjøres

```shell
python -m venv <name-of-venv>
```

**Notat:** Det kan være dere må skrive `python3` og `pip3` istedenfor `python` og `pip`. Dette skjer ofte dersom både Python 2 og 3 er installert på datamaskinen og Anaconda ikke er det. Vi kommer til å bruke kun `python` og `pip` i instruksjonene og innholdet i dette faget.

Denne kommandoen vil lage en mappe som heter `<name-of-venv>` og lagre den der hvor kommandoen blir kjørt. Det er vanlig å kalle den for `.venv`, der `.` i starten markerer at det vil være en skjult mappe. Dette kan dog bli vanskelig å håndtere dersom dere har mange virtuelle miljøer, så vi anbefaler at dere kaller den for noe beskrivende, for eksempel `in1160-venv`.

Dere kan aktivere miljøet med følgende:

- **MacOS / Linux:**: `source <name-of-venv>/bin/activate`. Her vil kommandoen `source` kjøre linjer i terminalen, mappen `bin/` inneholder "binaries" (executables) og `activate` er et shell script for å starte miljøet.
- **Windows OS**:
  - CMD: `<name-of-venv>\Scripts\activate`
  - Powershell: `.\<name-of-venv>\Scripts\activate`

Dersom dere har konfigurert terminalen deres kan det være at prompten deres nå viser hvilket miljø dere har aktivert.

Dere kan nå installere pakkene som trengs i faget. Man kan installere pakker Python med `pip install <name-of-package>`. Det er vanlig å lage en fil kalt `requiremnets.txt` som inneholder alle pakkene som trengs i et prosjekt. Da kan man installere alt på én gang med `pip install -r requirements.txt`, der `-r` står for recursive. Dette vil nok ta noen minutter om dere ikke har installert disse pakke før

For å deaktivere et virtuelt miljø, kjør `deactivate` i terminalen.

### Integrering med VS Code

VS Code har funksjonalitet for å håndtere miljøer og installasjoner. Først åpne hele mappen til dette repositoriet i VS Code. Deretter trykk på `ctrl / cmd` + `shift` + `p` og søk og velg `Python: Select interpreter`. Her får dere valg mellom de tilgjengelige Python-miljøene. Velg miljøet dere har laget. Slik vil VS Code forstå akkurat hvilke pakker dere har tilgjengelig i miljøet deres.

Dette gjør at linteren deres kan automatisk forstå om dere importerer noe fra en pakke dere ikke har (for eksempel `import numpy as np` vil få en strek og problem relatert til seg dersom `numpy` ikke er installert). Det vil også hjelpe IntelliSensen til Python med å vise hvilke funksjoner som er tilgjengelig i hvilke pakker og hvilke argumenter de tar. VS Code kan også aktivere riktig miljø i terminalen automatisk når dere åpner prosjektet (mappen).

## (Bonus) Andre virtuelle miljøer og pakkebehandlere

Vi har her sett på Pythons standard virtuelle miljø og pakkebehandler `pip`, som er det som installerer og håndterer installasjoner. Fellesskapet rundt Python har laget mange alternativer som har flere fordeler over standardmiljøene og pakkebehandleren. Siden det uansett er viktig å kunne standardisere måten å gjøre ting på er det det vi bruker i dette oppsettet, og det er ikke pensum å vite mer om virtuelle miljøer og pakkebehandlere i dette faget. Allikevel kan det være nyttig å vite litt om noen populære alternativer dersom dere fortsetter å programmere i Python.

**Anaconda**: For mer funksjonalitet og prosjekter som bruker store installasjoner (som pakker for nevrale nettverk) er det populært å bruke *Anaconda*. Anaconda gjør det mulig å enkelt bytte mellom forskjellige versjoner av Python, kommer med en pakkebehandler `conda` som er nøyere på å finne konflikter mellom pakker og mye mer. Dersom dere bruker Windows kan det være praktisk å bruke terminalen som følger med, `Anaconda prompt`, da denne inneholder kommandoer som likner mer på Linux. De virtuelle miljøene laget med `conda` er lagret globalt, så man slipper å ha mapper med installasjoner spredt rundt om i forskjellige mapper. Nedsiden med Anaconda er at det tar større plass, er noe tyngre og kjøre og kommer med en del funksjonalitet man kanskje ikke trenger.

**uv**: Et mye raskere alternativ til `pip`, som nylig har hatt en kraftig økning i popularitet, er `uv`. Den største fordelen med `uv` er at den er mye raskere enn `pip`, rundt 10–100x ganger raskere. Det har også sin egen måte å definere Python-prosjekter og virtuelle miljøer på. Det er laget av Astral software, som er de samme utviklerne bak formatereren og linteren `Ruff` som vi sterkt anbefaler å bruke. Gitt popularitetskurven til `uv` er det sannsynlig at den kan ta over som en standard pakkebehandler i fremtiden, men er foreløpig i et ganske tidlig stadium.
