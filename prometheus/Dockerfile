FROM prom/prometheus:v3
USER root

COPY alerts.yml /etc/prometheus/alerts.yml
COPY entrypoint.sh /entrypoint.sh


RUN sed -i 's/\r$//' /entrypoint.sh && chmod +x /entrypoint.sh


RUN mkdir -p /prometheus && chown -R nobody:nobody /prometheus

EXPOSE 9090

ENTRYPOINT ["/entrypoint.sh"]


FROM prom/prometheus:v3.4.0

USER root

# Copy configuration files
COPY alerts.yml /etc/prometheus/alerts.yml
COPY entrypoint.sh /entrypoint.sh

# Fix line endings and make executable (now works because this image has /bin/sh)
RUN sed -i 's/\r$//' /entrypoint.sh && chmod +x /entrypoint.sh

# Create prometheus directory with correct permissions
RUN mkdir -p /prometheus && chown -R nobody:nobody /prometheus

# Switch back to non-root user
USER nobody

EXPOSE 9090

ENTRYPOINT ["/entrypoint.sh"]
CMD ["--config.file=/etc/prometheus/prometheus.yml", \
     "--storage.tsdb.path=/prometheus", \
     "--web.console.libraries=/usr/share/prometheus/console_libraries", \
     "--web.console.templates=/usr/share/prometheus/consoles"]