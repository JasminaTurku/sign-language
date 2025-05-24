## Sign-language
Aplikacija omogućava osobama oštećenog sluha lakšu komunikaciju sa osobama koje nemaju taj problem. Razvijena je u Pythonu, koristeći biblioteku MediaPipe, koja omogućava praćenje pokreta prstiju u realnom vremenu, koristeći algoritme za prepoznavanje ključnih tačaka na ruci.

## Sadržaj
### Problem koji aplikacija rešava?
### Struktura projekta
### Media pipe biblioteka
### Realizacija projekta
### Koje kompanije i projekti su pokusali da reše problem
### Instalacija

## Problem koji aplikacija rešava?
Glavni problem sa kojim se suočavaju osobe oštećenog sluha je način na koji komuniciraju sa ljudima koji ne poznaju znakovni jezik. U Srbiji je ovaj problem posebno izražen, jer prema podacima Saveza gluvih i nagluvih Srbije, oko 70.000 osoba koristi srpski znakovni jezik kao svoj prvi jezik. Nažalost, u Srbiji postoji oko 30 tumača znakovnog jezika za 70.000 ljudi. Nedostatak tumača ozbiljno otežava svakodnevni život osoba koje koriste znakovni jezik. Takođe, usluga tumača je uglavnom dostupna samo u kritičnim situacijama, dok su svakodnevne potrebe ove zajednice često zapostavljene.
Ova aplikacija ima za cilj da pomogne u prevazilaženju tog problema. Korišćenjem kamere, korisnici mogu pokazati znak, a aplikacija će prepoznati i prikazati odgovarajuće slovo. Takođe, postoji opcija u kojoj korisnik može izgovoriti slovo putem mikrofona, a aplikacija će to slovo pretvoriti u odgovarajući znak.

## create virtual environment

```
python -m venv .venv
```

## Select interpreter

```
ctrl+shift+p -> python select interpreter
```


## Dependenices

### Generate requirements.txt from an existing environment:

```
pip freeze > requirements.txt
```

### Install libraries from the requirements.txt file

```
pip install -r requirements.txt
```
