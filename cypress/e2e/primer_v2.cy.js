describe("The Primer v2: İstemci Çıkarımı, Oyuncu A/B Simülasyonu ve MLOps HUD Testleri", () => {
  beforeEach(() => {
    cy.visit("/");
    cy.get('[data-cy="runtime-status"]', { timeout: 15000 })
      .should("contain.text", "ONNX Runtime Hazır");
  });

  it("1. Model Sözleşmesi ve Çalışma Zamanı Başarıyla Yüklenmeli", () => {
    cy.get('[data-cy="runtime-status"]').should("have.class", "success");
    cy.get('[data-cy="parity-badge"]').should("be.visible");
  });

  it("2. Oyuncu Davranışsal Telemetrisi (Karar Değiştirme) Canlı Takip Edilmeli", () => {
    cy.get('[data-cy="disp-changes"]').should("contain.text", "0");
    cy.get('[data-cy="choice-easy"]').check({ force: true });
    cy.get('[data-cy="disp-changes"]').should("contain.text", "1");
  });

  it("3. Oyuncu A Senaryosu (Tereddütlü) Çalıştırıldığında İpucu Tetiklenmeli ve HUD Güncellenmeli", () => {
    // Oyuncu A simülasyon butonuna bas
    cy.get('[data-cy="demo-player-a"]').click();

    // MLOps HUD üzerindeki P değerini kontrol et (>= 0.75 olmalı)
    cy.get('[data-cy="p"]', { timeout: 10000 })
      .invoke("text")
      .then((val) => {
        const p1 = parseFloat(val);
        expect(p1).to.be.greaterThan(0.75);
      });

    // Dinamik iskele kutusu görünür olmalı
    cy.get('[data-cy="hint-banner"]').should("be.visible");
    cy.get('[data-cy="flow-banner"]').should("not.be.visible");
  });

  it("4. Oyuncu B Senaryosu (Uzman) Çalıştırıldığında İpucu Engellenmeli ve Akış Derinleşmeli", () => {
    // Oyuncu B simülasyon butonuna bas
    cy.get('[data-cy="demo-player-b"]').click();

    // MLOps HUD üzerindeki P değerini kontrol et (< 0.20 olmalı)
    cy.get('[data-cy="p"]', { timeout: 10000 })
      .invoke("text")
      .then((val) => {
        const p1 = parseFloat(val);
        expect(p1).to.be.lessThan(0.20);
      });

    // Standart / Uzman akış kutusu görünür olmalı, ipucu gizlenmeli
    cy.get('[data-cy="flow-banner"]').should("be.visible");
    cy.get('[data-cy="hint-banner"]').should("not.be.visible");
  });
});
