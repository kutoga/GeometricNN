Was soll gemacht und gezeigt werden?

- Multiplikative Ableitungen (vor allem die geometrische) eignen sich, um neuronale Netz zu optimieren, insbesondere multiplikative neuronale Netze
- Durch die Idendität $f'(x)=f(x)\log(f^{\cup}(x))$ kann man multiplikative Funktionen in neuronalen Netzen trivial ableiten
  - Dies bezieht sich insbesondere auch auf einzelne Layer
  - Hiermit kann man auch "normale" Optimierungsmethoden (RMSprop etc.) nutzen, um multiplikative zu optimieren

Beispiele:
- Alles soll anhand eines XOR-Netzwerks gezeigt werden
- Die zweite Angabe, nämlich dass man Funktionen wie $\prod x_i^{w_i}$ trivial ableiten kann, indem man die genannte Identität nutzt, soll analytisch gezeigt werden und von Hand nachgerechnet werden.
  - Das Vereinfacht die Ableitung von multiplikativen Layern enorm und macht diese so viel nützlicher
