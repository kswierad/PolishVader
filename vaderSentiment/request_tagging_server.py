import requests

data = '''Piją, jadą i okazują się niewinni. Ministerstwo Zbigniewa Ziobry chce to ukrócić Nietrzeźwy kierowca bierze udział w wypadku, a potem twierdzi, że pił dopiero po kraksie – policja coraz częściej spotyka się z takimi sytuacjami. Resort sprawiedliwości chce je całkowicie wyeliminować. Jeśli sprawca wypadku sięgnie po butelkę, zanim zostanie poddany badaniu alkomatem, będzie traktowany tak, jakby spowodował go w stanie nietrzeźwości lub zbiegł z miejsca zdarzenia – zapowiada ministerstwo Zbigniewa Ziobry. W ten sposób resort zamierza walczyć z kierowcami, którzy przed sądem zarzekają się, że pili dopiero po zdarzeniu. Żeby ukoić nerwy. Co jakiś czas zdarzają się sytuacje, w których oskarżony próbuje w ten sposób, mówiąc kolokwialnie, zrobić sąd w konia. Powołujemy wtedy biegłych, ale nie zawsze są oni w stanie jednoznacznie określić, jakie stężenie miał kierujący w momencie wypadku – przyznaje sędzia Rafał Lisak z Sądu Okręgowego w Krakowie. Adam Reza z Polskiego Stowarzyszenia Biegłych Sądowych do Spraw Wypadków Drogowych tłumaczy, że wyniki badań retrospektywnych są obarczone dużą niepewnością. Zwłaszcza gdy kierowca faktycznie pije po wypadku, by maksymalnie utrudnić ustalenie stanu trzeźwości w określonym czasie. Dodatkowo zdarza się, że wychodzi z samochodu i sięga po butelkę tak, by widziało to jak najwięcej osób. W ten sposób uprawdopodabnia swoją linię obrony. Nie jest to sposób dający stuprocentową skuteczność, ale tam, gdzie opinie biegłych są niejednoznaczne, wątpliwości interpretuje się na korzyść oskarżonego – zgodnie z art. 5 par. 2 k.p.k. Sądy zbyt często dają się nabrać na takie absurdalne tłumaczenia – uważa Marcin Warchoł, wiceminister sprawiedliwości. I zapowiada zmianę przepisów. Nieważne, czy sprawca wypadku pił przed zdarzeniem czy po nim – wszystko będzie okolicznością zaostrzającą odpowiedzialność karną za spowodowanie wypadku, katastrofy w komunikacji oraz sprowadzenie bezpośredniego jej niebezpieczeństwa. Ponadto w kodeksie drogowym pojawi się obowiązek powstrzymania się od picia alkoholu lub zażywania narkotyków do czasu przyjazdu policji. Ma on dotyczyć wszystkich uczestników wypadku, a nie tylko jego sprawcy. Co ciekawe, zakaz nie będzie obowiązywał kierowców biorących udział w kolizji.'''
response = requests.post('http://localhost:9200/?output_format=conll', data=data.encode('utf-8'))

if response.status_code != 200:
    print("requesting server failed")

string_response = str(response.content, 'utf8')
words = string_response.split("\\")
words_splitted = words[0].split()
# print(words_splitted)
list_of_skipped_words = ["disamb", "none", "space", "conj", "interp", "newline", "comp", "qub", "adv", "pred"]
for word in words_splitted:
    if word in list_of_skipped_words or ":" in word:
        words_splitted.remove(word)

# second iteration, because the first one doesn't remove "dismab" words
for word in words_splitted:
    if word in list_of_skipped_words or ":" in word:
        words_splitted.remove(word)

# getting every second element as those are tagged words
words_splitted = words_splitted[1::2]

print(words_splitted)
