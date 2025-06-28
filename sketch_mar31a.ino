#define MOTOR_PIN 9  // Pin pentru controlul motorului

const int analogPin = A0;  // Definim pinul A0
int valoareAnalog = 0;     // Variabilă pentru stocarea citirii
int stare = 0;
int i = 0;
int medie = 0;
int calibrare1 = 0;
int calibrare2 = 0;
int calibrare3 = 0;
int titrare = 0;
int suma = 0;
unsigned long startTime = 0;
unsigned long maxTime = 90000;  // 90 secunde = 90000 milisecunde
unsigned long maxTimeTitrare = 3600000;  // 60 minute = 24000 secunde = 24000000 milisecunde

void setup() {
    pinMode(MOTOR_PIN, OUTPUT); // Setează pinul motorului ca ieșire
    digitalWrite(MOTOR_PIN, LOW); // Asigură-te că motorul este oprit la pornire
    Serial.begin(9600); // Inițializăm comunicația serială
}

void loop() {
    // Așteaptă un semnal de start sau stop de la Python
    if (Serial.available() > 0) {
        String mesaj = Serial.readStringUntil('\n');
        mesaj.trim();  // Elimină spațiile goale

        if (mesaj == "Calibrare1") {
            stare = 1;
            startTime = millis();  // Pornim timerul
        }
        else if (mesaj == "Calibrare2") {
            stare = 2;
            startTime = millis();  // Pornim timerul
        }
        else if (mesaj == "Calibrare3") {
            stare = 3;
            startTime = millis();  // Pornim timerul
        }
        else if (mesaj == "Titrare") {
            stare = 4;
            startTime = millis();  // Pornim timerul
        }
    }

    if (stare == 1 && (calibrare1 == 0)) {
        calibrare1 = 1;
        while (millis() - startTime < maxTime) {
            valoareAnalog = analogRead(analogPin); // Citim valoarea de la A0
            medie += valoareAnalog;
            Serial.println(valoareAnalog); // Afișăm valoarea în Serial Monitor
            delay(5000);  // Pauză de 5 secunde

            i = (i + 1) % 10;
            if (i == 0) {
                medie /= 10;
                Serial.print("Valoare finala pentru Calibrare 1 (pH 6.86): ");
                Serial.println(medie);  // Afișează valoarea medie
                calibrare1 = 2;
                break;
            }
        }

        if (millis() - startTime >= maxTime) {
            Serial.println("Timpul pentru Calibrarea 1 a expirat. Poti trece la urmatoarea etapa.");
        }
    }

    if (stare == 2 && (calibrare2 == 0)) {
        calibrare2 = 1;
        while (millis() - startTime < maxTime) {
            valoareAnalog = analogRead(analogPin); // Citim valoarea de la A0
            medie += valoareAnalog;
            Serial.println(valoareAnalog); // Afișăm valoarea în Serial Monitor
            delay(5000);  // Pauză de 5 secunde

            i = (i + 1) % 10;
            if (i == 0) {
                medie /= 10;
                Serial.print("Valoare finala pentru Calibrare 2 (pH 4.01): ");
                Serial.println(medie);  // Afișează valoarea medie
                calibrare2 = 2;
                break;
            }
        }

        if (millis() - startTime >= maxTime) {
            Serial.println("Timpul pentru Calibrarea 2 a expirat. Poti trece la urmatoarea etapa.");
        }
    }

    if (stare == 3 && (calibrare3 == 0)) {
        calibrare3 = 1;
        while (millis() - startTime < maxTime) {
            valoareAnalog = analogRead(analogPin); // Citim valoarea de la A0
            medie += valoareAnalog;
            Serial.println(valoareAnalog); // Afișăm valoarea în Serial Monitor
            delay(5000);  // Pauză de 5 secunde

            i = (i + 1) % 10;
            if (i == 0) {
                medie /= 10;
                Serial.print("Valoare finala pentru Calibrare 3 (pH 9.21): ");
                Serial.println(medie);  // Afișează valoarea medie
                calibrare3 = 2;
                break;
            }
        }

        if (millis() - startTime >= maxTime) {
            Serial.println("Timpul pentru Calibrarea 3 a expirat. Poti trece la urmatoarea etapa.");
        }
    }

    if (stare == 4 && (titrare == 0)) {
        titrare = 1;
        valoareAnalog = analogRead(analogPin); // Citim valoarea de la A0
        Serial.println(valoareAnalog); // Afișăm valoarea în Serial Monitor
        while (millis() - startTime < maxTimeTitrare) {
            digitalWrite(MOTOR_PIN, HIGH); // Pornește motorul
            delay(6400);  // Menține motorul pornit timp de 6.4 secunde
        
            digitalWrite(MOTOR_PIN, LOW); // Oprește motorul
            delay(13600); // Așteaptă 13.6 secunde înainte de următoarea activare

            suma = 0; // Resetează suma înainte
            for (i = 0; i < 10; i++) {
                suma += analogRead(analogPin); // Adună fiecare valoare
                delay(1000); // Așteaptă 1 sec între măsurători
            }
            valoareAnalog = suma / 10; // Media
            Serial.println(valoareAnalog); // Afișăm valoarea în Serial Monitor
        }

        if (millis() - startTime >= maxTimeTitrare) {
            Serial.println("Timpul pentru Titrare a expirat. Ati terminat masurarea.");
        }
    }
}
