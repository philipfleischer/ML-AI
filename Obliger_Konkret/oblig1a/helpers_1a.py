import importlib
import string

REQUIREMENTS = {
    "numpy": "numpy",
    "matplotlib": "matplotlib",
    "pandas": "pandas",
    "ipykernel": "ipykernel",
    "sklearn": "scikit-learn",
    "seaborn": "seaborn",
    "openml": "openml",
    "yaml": "pyyaml",
    "IPython": "ipython",
    "plotly": "plotly",
    "gymnasium": "gymnasium",
    "datasets": "datasets",
}


def sanity_check():
    """
    Laster inn alle pakker som trengs for IN1160-kurset
    og sjekker om de fungerer som forventa.
    """
    ok = True

    print("=== Sjekker pakker ===")
    for import_name, package_name in REQUIREMENTS.items():
        try:
            importlib.import_module(import_name)
            version = importlib.metadata.version(package_name)
            print(f"[OK] {package_name} ({version})")
        except Exception as e:
            print(f"[ERROR] Kunne ikke importere {package_name}. Sjekk installasjon --> {e}")
            ok = False

    try:
        importlib.import_module("gymnasium.envs.toy_text")
        print("[OK] gymnasium[toy-text]")
    except Exception:
        print("[!] gymnasium[toy-text] er ikke installert")
        ok = False

    print("====================")
    if ok:
        print("Alt fungerer som det skal!")
    else:
        print("Noe er kanskje feil med installasjonen av pakkene. Se feilmeldinger over.")


def get_norwegian_stopwords():
    """
    Gir en liste med norske stoppord, inkludert tegnsetting.
    """
    stopwords = [
        "og", "i", "jeg", "det", "at", "en", "et", "den", "til", "er", "som", "på", "de",
        "med", "han", "av", "ikke", "ikkje", "der", "så", "var", "meg", "seg", "men", "ett",
        "har", "om", "vi", "min", "mitt", "ha", "hadde", "hun", "nå", "over", "da", "vart",
        "ved", "fra", "du", "ut", "sin", "dem", "oss", "opp", "man", "kan", "hans", "hvor",
        "eller", "hva", "skal", "selv", "sjøl", "her", "alle", "vil", "bli", "ble", "blei",
        "blitt", "kunne", "inn", "når", "være", "kom", "noen", "noe", "ville", "dere", "som",
        "deres", "kun", "ja", "etter", "ned", "skulle", "denne", "for", "deg", "si", "sine",
        "sitt", "mot", "å", "meget", "hvorfor", "dette", "disse", "uten", "hvordan", "ingen",
        "din", "ditt", "blir", "samme", "hvilken", "hvilke", "sånn", "inni", "mellom", "vår",
        "hver", "hvem", "vors", "hvis", "både", "bare", "enn", "fordi", "før", "mange", "også",
        "slik", "vært", "være", "båe", "begge", "siden", "dykk", "dykkar", "dei", "deira",
        "deires", "deim", "di", "då", "eg", "ein", "eit", "eitt", "elles", "honom", "hjå",
        "ho", "hoe", "henne", "hennar", "hennes", "hoss", "hossen", "ikkje", "ingi", "inkje",
        "korleis", "korso", "kva", "kvar", "kvarhelst", "kven", "kvi", "kvifor", "me", "medan",
        "mi", "mine", "mykje", "no", "nokon", "noka", "nokor", "noko", "nokre", "si", "sia",
        "sidan", "so", "somt", "somme", "um", "upp", "vere", "vore", "verte", "vort", "varte"
    ]
    stopwords.extend(list(string.punctuation))

    return stopwords
