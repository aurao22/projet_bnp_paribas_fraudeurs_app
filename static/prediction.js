// contient les articles de presse, qui doivent être 
// gardés en mémoire même après affichage du graphique
var news_data;

// Palette de couleurs utilisée par tous les graphiques
var colors = ["#1D507A", "#2F6999", "#66A0D1", "#8FC0E9", "#4682B4"];

function load_prediction(){
    // Chargement des articles de presse
    $.ajax({
        url: "/api/predictions",
        success: display_prediction
    });
}


function display_prediction(result){
    console.log(result)

    data = result["data"];
    console.log(data)

    // TODO : voir comment est constitué la réponse
    var pred = data["prediction"];

    // Affichage du résultat dans la page web
    var div = $("#resultat_prediction").html("");
    div.append("<p>Voici le résultat de la prédiction :  "+pred+" (0 non frauduleux, 1 frauduleux)</p>");
}

function addRow() {
    var table = document.getElementById("cart_table");
    var row = table.insertRow(-1);

    var cell1 = row.insertCell(0);
    cell1.innerHTML = "nb_row";
    
    var cell2 = row.insertCell(1);
    cell2.innerHTML = "<input id='codeX' name='codeX' value=''/>";
    
    var cell2 = row.insertCell(2);
    cell2.innerHTML = "<input id='modelX' name='modelX' value=''/>";

    var cell2 = row.insertCell(3);
    cell2.innerHTML = "<input id='itemX' name='itemX' value=''/>";

    var cell2 = row.insertCell(4);
    cell2.innerHTML = "<input id='nbX' name='nbX' value=''/>";
    
    var cell2 = row.insertCell(5);
    cell2.innerHTML = "<input id='cashX' name='cashX' value=''/>";

    var cell2 = row.insertCell(6);
    cell2.innerHTML = "<input id='makeX' name='makeX' value=''/>";
  }


