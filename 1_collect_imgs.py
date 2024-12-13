import os
import cv2
import time

# Podešavanje foldera za čuvanje podataka
DATA_DIR = './data'
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

# Broj slika po slovu
dataset_size = 100  # Broj slika po slovu

# Aktiviraj kameru
cap = cv2.VideoCapture(0)

ESC = 27
ENTER = 13

try:
    while True:
        # Prikaz poruke za unos slova preko kamere
        selected_letter = None
        while True:
            ret, frame = cap.read()
            if not ret:  # Ako kamera ne učita frejm
                print("Greška pri učitavanju frejma kamere. Pokušavam ponovo...")
                continue

            cv2.putText(frame, "Unesite slovo ili ESC za kraj programa",
                        (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.imshow('frame', frame)

            key = cv2.waitKey(1) & 0xFF
            if 65 <= key <= 90 or 97 <= key <= 122:  # Ako je slovo
                selected_letter = chr(key).upper()
                break
            elif key == ESC:  
                print("Završetak programa.")
                break

        if not selected_letter:
            break

        letter = selected_letter
        print(f"Uneto slovo: {letter}")

        # Kreiraj direktorijum za dato slovo
        letter_dir = os.path.join(DATA_DIR, letter)
        if not os.path.exists(letter_dir):
            os.makedirs(letter_dir)

        print(f'Prikupljanje podataka za slovo: {letter}')

        # Prikaz poruke i čekanje korisnika da pritisne ENTER za početak
        while True:
            ret, frame = cap.read()
            if not ret:  # Ako kamera ne učita frejm
                print("Greška pri učitavanju frejma kamere. Pokušavam ponovo...")
                continue

            cv2.putText(frame, f'Slovo "{letter}": ENTER za snimanje ili ESC za kraj programa',
                        (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.imshow('frame', frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ENTER: 
                break
            elif key == ESC:  
                print("Prekid unosa za slovo. Vratite se na unos novog slova.")
                break

        if key == ESC:  # Provera za ESC iz spoljnog petlje
            continue

        # Prikupljanje podataka
        counter = 0
        while counter < dataset_size:
            ret, frame = cap.read()
            if not ret:  # Proveri učitavanje
                print("Greška pri učitavanju frejma kamere. Preskačem...")
                continue

            cv2.putText(frame, f'Prikupljanje: {letter} ({counter + 1}/{dataset_size})',
                        (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.imshow('frame', frame)

            # Pauza da se osigura da kamera dobije dovoljno vremena
            time.sleep(0.1)

            # Sačuvaj sliku
            cv2.imwrite(os.path.join(letter_dir, f'{counter}.jpg'), frame)
            counter += 1

            # Proveri da li je korisnik pritisnuo "ESC" za prekid
            if cv2.waitKey(1) & 0xFF == 27:
                print("Prekid prikupljanja...")
                break
finally:
    cap.release()
    cv2.destroyAllWindows()
