import os
import cv2
import time  # Dodaj za kontrolu pauze

# Podešavanje foldera za čuvanje podataka
DATA_DIR = './data'
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

# Broj klasa i veličina dataset-a
number_of_classes = 26 # Za 30 slova
dataset_size = 100  # Broj slika po slovu

# Srpska azbuka
letters = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M","N", "O","P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z"]

# Aktiviraj kameru
cap = cv2.VideoCapture(0)

for j, letter in enumerate(letters):
    # Kreiraj direktorijum za svako slovo
    letter_dir = os.path.join(DATA_DIR, letter)
    if not os.path.exists(letter_dir):
        os.makedirs(letter_dir)

    print(f'Prikupljanje podataka za slovo: {letter}')

    # Priprema za prikupljanje podataka
    while True:
        ret, frame = cap.read()
        if not ret:  # Ako kamera ne učita frejm
            print("Greška pri učitavanju frejma kamere. Pokušavam ponovo...")
            continue

        cv2.putText(frame, f'Spremni za "{letter}"? Pritisnite "Q" za početak!',
                    (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.imshow('frame', frame)
        if cv2.waitKey(25) == ord('q'):
            break

    # Prikupljanje podataka
    counter = 0
    while counter < dataset_size:
        ret, frame = cap.read()
        if not ret:  # Proveri učitavanje
            print("Greška pri učitavanju frejma kamere. Preskačem...")
            continue

        cv2.putText(frame, f'Prikupljanje: {letter} ({counter + 1}/{dataset_size})',
                    (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.imshow('frame', frame)

        # Pauza da se osigura da kamera dobije dovoljno vremena
        time.sleep(0.1)

        # Sačuvaj sliku
        cv2.imwrite(os.path.join(letter_dir, f'{counter}.jpg'), frame)
        counter += 1

        # Proveri da li je korisnik pritisnuo "ESC" za prekid
        if cv2.waitKey(25) == 27:
            print("Prekid prikupljanja...")
            break

cap.release()
cv2.destroyAllWindows()
