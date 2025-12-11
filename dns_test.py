# dns_test.py
import socket, time
host = "tds-llm-analysis.s-anand.net"
for i in range(1, 101):
    try:
        addrs = socket.getaddrinfo(host, 443)
        print(i, "OK", addrs[:2])
    except Exception as e:
        print(i, "ERR", repr(e))
    time.sleep(0.3)
