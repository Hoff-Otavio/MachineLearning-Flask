
// Atualiza diniamicamente o gráfico.
function updateChart(imgData) {
    var resultContainer = document.getElementById('result-container');
    
    // Cria uma nova imagem para o gráfico
    var img = new Image();
    img.src = 'data:image/png;base64,' + imgData;
    
    // Remove qualquer conteúdo existente na div
    while (resultContainer.firstChild) {
        resultContainer.removeChild(resultContainer.firstChild);
    }

    // Adiciona a nova imagem à div
    resultContainer.appendChild(img);
}


