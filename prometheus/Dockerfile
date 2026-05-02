FROM prom/prometheus:v3-distroless
USER root

COPY alerts.yml /etc/prometheus/alerts.yml
COPY entrypoint.sh /entrypoint.sh


RUN sed -i 's/\r$//' /entrypoint.sh && chmod +x /entrypoint.sh


RUN mkdir -p /prometheus && chown -R nobody:nobody /prometheus

EXPOSE 9090

ENTRYPOINT ["/entrypoint.sh"]