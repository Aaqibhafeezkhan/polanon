"""First names from the PESEL register that the hand-written lists in names_pl lack.

Every first name borne by at least 1 000 people in Poland, from the lists of first names
in the PESEL register as of 22 January 2025 that the Ministry of Digital Affairs publishes on
dane.gov.pl under CC0 1.0 (dataset 1667). With them the first names polanon knows without a
language model cover about 97% of the people in the register, among them the Ukrainian and
Belarusian names that Poland's residents bear, written as the register writes them.

AMBIGUOUS lists the folded forms of these names that Polish text also writes as ordinary
words, in lowercase, or that are small English words: names_pl treats them as it treats
"Kalina" or "Wiktoria". A form counts as a word when KPWr, the Polish rows of ai4privacy's
OpenPII or the benchmark samples write it in lowercase twice or more, and by review: Sylwester,
Malina, Melisa, Ida ("idę") and Alex ("alexa") among others. Seven names that are English words
in tools and logs are left out: Max, Mark, Frank, Roger, Pascal, Christian and Aurora.

Generated once from the register; a name added by hand belongs in the lists of names_pl.
"""

MALE = """
    Aaron Adolf Ahmed Ahmet Albin Aleh Aleksandr Aleksei Aleksy Alex Alexander Alexandr
    Alexandru Alfons Ali Aliaksandr Aliaksei Alojzy Amadeusz Amir Anatol Anatoli Anatolii
    Anatoliy Andreas Andrei Andrew Andrii Andriy Anthony Anton Antonio Ariel Arnold Aron Arsen
    Arsenii Artem Arthur Artsiom Augustyn Beniamin Benjamin Bohdan Brajan Brayan Brian Bruno
    Brunon Carlos Christopher Cristian Damir Daniil Danil Danyil Danylo David Davit Davyd
    Demian Denis Dennis Denys Diego Dionizy Dmitrii Dmytro Dominic Dorian Dumitru Dzianis
    Dzmitry Edgar Eduard Edwin Eliasz Eligiusz Emilian Eric Erik Erwin Ewald Fedir Felix
    Ferdynand Francesco George Gerhard Gheorghe Ginter Giorgi Helmut Henadzi Hennadii Henry
    Heorhii Herbert Herman Hlib Horst Hryhorii Hugo Ievgen Ihar Ihor Ilia Illia Ilya Ion Irakli
    Iurii Ivan Iwo Izydor Jacenty Jacob James Jaromir Jeremi Jeremiasz John Jonasz Jonatan
    Jonathan Jordan Jorge Jose Joseph Joshua Juan Kanstantsin Kasjan Kasper Kevin Kewin Kiril
    Kirill Kiryl Klemens Kordian Korneliusz Kosma Kostiantyn Kryspin Ksawier Kyryl Kyrylo Lasha
    Lechosław Leo Leonardo Leonid Leopold Lesław Lev Levan Liam Liubomyr Luca Lucas Luis Luka
    Lukas Makar Maksim Maksym Mamuka Manfred Manuel Marceli Marco Marko Markus Martin Matteo
    Matthew Matvii Maxim Maximilian Mehmet Michael Mieszko Mihail Mikalai Mikhail Mikita Milan
    Miron Mohamed Mohammad Mohammed Muhammad Mustafa Mykhailo Mykhaylo Mykola Mykyta Myron
    Myroslav Nathan Nazar Nazarii Nicholas Nicolae Nicolas Nikita Nikolas Noah Oktawian Oleg
    Oleh Oleksandr Oleksii Oleksiy Oliver Olivier Omar Orest Oscar Ostap Patrick Patrycjusz
    Paul Pavel Pavlo Peter Petro Philip Platon Radomir Rafael Rajmund Raman Richard Roch Rodion
    Roland Romuald Rostyslav Rudolf Ruslan Rustam Ryan Samir Samuel Semen Sergei Sergey Serghei
    Sergii Sergiu Sergiusz Sergiy Serhii Serhiy Siarhei Simon Stanislau Stanislav Stepan
    Sviatoslav Sylwester Taras Teofil Thomas Timur Tymofii Tymur Tytus Uladzimir Uladzislau
    Vadim Vadym Vadzim Valentyn Valerii Valery Vasile Vasili Vasyl Viachaslau Viacheslav Victor
    Viktar Viktor Vincent Vitali Vitalii Vitaliy Vladimir Vladislav Vladyslav Volodymyr
    Walerian Walter Wawrzyniec Werner Wilhelm William Wit Xavier Yahor Yan Yaroslav Yauhen
    Yauheni Yehor Yevhen Yevhenii Yurii Yuriy Yury Zakhar Ziemowit Zurab Zygfryd
"""

FEMALE = """
    Abigail Adela Adelajda Adelina Aksana Ala Albina Alena Alesia Alexandra Aliaksandra Alice
    Alicia Alisa Alla Alona Amanda Amina Ana Anastasia Anastasiia Anastasiya Andrea Andriana
    Andżelika Anetta Angela Angelina Anhelina Anika Antonia Anzhela Anzhelika Ariana Arina
    Arleta Arletta Bernadeta Bernadetta Bernardyna Bianca Bogna Bohdana Bożenna Brygida Carmen
    Carolina Caroline Claudia Cristina Dajana Dalia Dana Dariia Darya Daryna Dina Dobrosława
    Donata Dżesika Edeltrauda Ekaterina Elena Eleonora Elfryda Elina Elizabeth Elizaveta Ella
    Elvira Emiliia Emily Erika Erna Estera Eugenia Eulalia Eva Evelina Faustyna Florentyna
    Gabriella Gaja Galina Galyna Ganna Gerda Gertruda Gizela Gloria Greta Halyna Hanka Hannah
    Helga Hildegarda Ida Ina Ines Inez Inga Inka Inna Irina Irmgarda Irmina Iryna Isabella
    Iuliia Ivanna Izabella Jagna Jaśmina Jennifer Jessica Jowita Julianna Kamilla Karyna
    Katerina Kateryna Katsiaryna Kaya Kazimiera Khrystyna Kira Klaudyna Klementyna Konstancja
    Kristina Krystsina Kseniia Kseniya Lara Larisa Larysa Lea Lesia Liana Lidiia Lila Lilia
    Liliia Liliya Lilla Lilly Lily Lina Linda Liubov Liudmila Liudmyla Livia Lizaveta Luba
    Luiza Lyubov Lyudmyla Maia Maiia Malina Manuela Marcjanna Margarita Marharyta Mariam
    Marietta Mariia Marika Marina Marita Mariya Marlena Martina Maryia Maryla Maryna Marzanna
    Marzenna Maya Melaniia Melisa Mia Michelle Mila Milana Mira Mirela Mirella Miriam Myroslava
    Nadiia Nadiya Nadzeya Nadzieja Naomi Nastassia Nataliia Nataliya Natallia Natasza Nel Nelia
    Nelly Nicol Nicola Nicole Nika Nikol Nikoletta Noemi Oksana Oktawia Oleksandra Olena Olesia
    Olha Olimpia Olivia Palina Pamela Patricia Pelagia Polina Raisa Rita Roksolana Roma
    Romualda Ruslana Ruta Sabrina Salomea Samanta Sarah Simona Snizhana Sofia Sofiia Sofiya
    Solomiia Sophia Sophie Stefaniia Stella Svetlana Sviatlana Svitlana Taisiia Tamila Tatiana
    Tatsiana Teodozja Teofila Tetiana Tetyana Tola Uliana Ulyana Valentina Valentyna Valeria
    Valeriia Valeryia Vanessa Varvara Vasylyna Vera Veranika Veronica Veronika Victoria
    Viktoria Viktoriia Viktoriya Viktoryia Violeta Violetta Vira Vita Vitaliia Vitalina Vlada
    Vladyslava Volha Walentyna Waleria Wanesa Wanessa Wiera Wioleta Wirginia Yana Yanina
    Yaroslava Yaryna Yelizaveta Yelyzaveta Yeva Yevheniia Yulia Yuliana Yuliia Yuliya Zenobia
    Zhanna Zinaida Zlata Zoe Zoia Zoja Zoriana Zyta
"""

AMBIGUOUS = """
    ale alex alexa alexem alexie alexowi any dalia dalie dalii dalio dana dane danie dano dany
    donate ida ide ido idy idzie inna inne inny lesie liana liane lianie liano liany lilia lina
    line linie lino liny lubie luke luki malina maline malinie malino maliny marina marine
    marinie marino mariny marzanna marzanne marzannie marzanno marzanny melisa melise melisie
    meliso melisy mie mila mile milo nadziei nadzieja nadzieje raise sylwester sylwestera
    sylwesterem sylwesterowi sylwesterze taras very wierze wita zoo zycie
"""
