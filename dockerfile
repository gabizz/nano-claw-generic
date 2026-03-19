FROM node:20-alpine AS builder
RUN apk add --no-cache git python3 make g++
WORKDIR /app
COPY ./package*.json ./
RUN npm install
COPY . .
RUN npm run build

FROM node:20-alpine
WORKDIR /app
COPY --from=builder /app/dist ./dist
COPY --from=builder /app/package*.json ./
RUN npm install --omit=dev
RUN mkdir -p /root/.nano-claw
COPY config.json /root/.nano-claw/config.json
EXPOSE 3000
CMD ["node", "dist/cli/index.js", "gateway"]
