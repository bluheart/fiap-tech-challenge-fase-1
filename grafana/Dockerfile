FROM grafana/grafana:main

# Copy provisioning configurations
COPY provisioning/ /etc/grafana/provisioning/

# Ensure correct permissions
USER root
RUN chown -R 472:472 /etc/grafana/provisioning/

# Switch back to Grafana user (472)
USER 472

EXPOSE 3000