window.addEventListener('DOMContentLoaded', event => {
    // Toggle the side navigation
    const sidebarToggle = document.body.querySelector('#sidebarToggle');
    if (sidebarToggle) {
        sidebarToggle.addEventListener('click', event => {
            event.preventDefault();
            document.body.classList.toggle('sb-sidenav-toggled');
            localStorage.setItem('sb|sidebar-toggle', document.body.classList.contains('sb-sidenav-toggled'));
        });
    }
});

// Créez la carte et définissez la vue initiale sur Rouen, avec un niveau de zoom élevé
var map = L.map('map').setView([49.4431, 1.0993], 12);

// Utilisez une couche de tuiles CartoDB Voyager
L.tileLayer('https://{s}.basemaps.cartocdn.com/rastertiles/voyager/{z}/{x}/{y}{r}.png', {
    attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors &copy; <a href="https://carto.com/attributions">CARTO</a>',
    subdomains: 'abcd',
    maxZoom: 19
}).addTo(map);

var redIcon = new L.Icon({
    iconUrl: 'static/imgs/tags/monu0.png',
    iconSize: [50, 50],
    iconAnchor: [12, 41],
    popupAnchor: [1, -34],
});

var restoIcon = new L.Icon({
    iconUrl: 'static/imgs/tags/resto1.png',
    iconSize: [50, 50],
    iconAnchor: [12, 41],
    popupAnchor: [1, -34],
});

// Fonction pour normaliser les noms de monuments pour les URL
function normalizeString(str) {
    return str.normalize("NFD") // décompose les caractères accentués
               .replace(/[̀-ͯ]/g, '') // enlève les diacritiques
               .replace(/[^a-zA-Z0-9\s]/g, '') // enlève les caractères non alphanumériques
               .replace(/\s+/g, '_') // remplace les espaces par des underscores
               .toLowerCase(); // convertit en minuscule
}

// Fonction pour tronquer le texte et ajouter un lien "Lire plus..."
function truncateText(text, wordLimit, url) {
    var words = text.split(' ');
    var truncatedText = text;
    var readMoreLink = '';

    if (words.length > wordLimit) {
        truncatedText = words.slice(0, wordLimit).join(' ') + '...';
        readMoreLink = ' <a href="' + url + '">Lire plus</a>';
    }

    return { truncatedText: truncatedText, readMoreLink: readMoreLink };
}

// Fonction pour créer le contenu de la popup
function createPopupContent(point, type) {
    var content = '<b>' + point.NOM + '</b><br>';
    if (point.PHOTO) {
        content += '<br><img src="' + point.PHOTO + '" alt="' + point.NOM + '" style="width:100px;height:auto;"> <br>';
    }

    // URL de la page du monument ou du restaurant utilisant le TSID
    var pageUrl = type === 'restaurant' ? '/restaurant/' + point.TSID : '/lieu/' + point.TSID;


    // Troncature du texte si nécessaire
    var { truncatedText, readMoreLink } = truncateText(point.DESCRIPTIF, 30, pageUrl);
    content += truncatedText;
    content += readMoreLink;

    content += `
        <br>
        <button class="custom-button" onclick="window.location.href='${pageUrl}'">Voir la page</button>
        <button class="custom-button" id="validate-visit-${point.LONGITUDE}-${point.LATITUDE}" onclick="validateVisit(${point.LONGITUDE}, ${point.LATITUDE}, \`${point.NOM}\`)">Valider la visite</button>
        <br>
        <button class="custom-button" onclick="redirectToItinerary(${point.LATITUDE}, ${point.LONGITUDE}, \`${point.NOM}\`, 'bike')">Itinéraire Vélo</button>
        <button class="custom-button" onclick="redirectToItinerary(${point.LATITUDE}, ${point.LONGITUDE}, \`${point.NOM}\`, 'bus')">Itinéraire Bus</button>
        <br>
        <button class="custom-button" id="feedback-button-${point.LONGITUDE}-${point.LATITUDE}" onclick="showFeedbackForm(${point.LONGITUDE}, ${point.LATITUDE}, \`${point.NOM}\`)">Émettre un feedback</button>
        <div id="feedback-form-${point.LONGITUDE}-${point.LATITUDE}" style="display:none;">
            <textarea id="feedback-text-${point.LONGITUDE}-${point.LATITUDE}" rows="3" cols="30" placeholder="Mettez votre feedback ici"></textarea>
            <br>
            <button class="custom-button" onclick="submitFeedback(${point.LONGITUDE}, ${point.LATITUDE}, \`${point.NOM}\`)">Soumettre</button>
        </div>

    `;

    return content;
}

// Récupérer les données depuis le serveur et ajouter des marqueurs sur la carte
fetch('/data')
    .then(response => response.json())
    .then(data => {
        data.lieux_visite.forEach(point => {
            if (!point.NOM || !point.LATITUDE || !point.LONGITUDE || !point.DESCRIPTIF) {
                return; // Skip this point if critical data is missing
            }
            L.marker([point.LATITUDE, point.LONGITUDE], { icon: redIcon })
                .bindPopup(createPopupContent(point, 'lieu'))
                .addTo(map);
        });

        data.restaurants.forEach(point => {
            if (!point.NOM || !point.LATITUDE || !point.LONGITUDE || !point.DESCRIPTIF) {
                return; // Skip this point if critical data is missing
            }
            L.marker([point.LATITUDE, point.LONGITUDE], { icon: restoIcon })
                .bindPopup(createPopupContent(point, 'restaurant'))
                .addTo(map);
        });
    })
    .catch(error => console.error('Error:', error));

// Fonction pour afficher le formulaire de feedback et cacher le bouton
function showFeedbackForm(longitude, latitude, nom) {
    document.getElementById(`feedback-form-${longitude}-${latitude}`).style.display = 'block';
    document.getElementById(`feedback-button-${longitude}-${latitude}`).style.display = 'none';
}

// Fonction pour soumettre un feedback
function submitFeedback(longitude, latitude, nom) {
    var text = document.getElementById(`feedback-text-${longitude}-${latitude}`).value;
    fetch('/feedback', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({ longitude: longitude, latitude: latitude, nom: nom, text: text })
    })
    .then(response => response.json())
    .then(data => {
        if (data.status === 'success') {
            alert('Feedback soumis avec succès, merci !');
            document.getElementById(`feedback-form-${longitude}-${latitude}`).style.display = 'none';
        } else {
            alert('Erreur lors de la soumission du feedback');
        }
    });
}

// Fonction pour valider la visite
function validateVisit(longitude, latitude, nom) {
    alert(`Visite validée pour ${nom}`);
}

function redirectToItinerary(destinationLat, destinationLng, name, mode) {
    // Redirige vers index.html avec l'adresse en paramètre
    window.location.href = `/index?destinationLat=${destinationLat}&destinationLng=${destinationLng}&name=${name}&mode=${mode}`;
}

// Fonction pour trouver un itinéraire
function findRoute(destinationLat, destinationLng, mode) {
    // Utiliser la géolocalisation pour obtenir la position actuelle de l'utilisateur
    if (navigator.geolocation) {
        navigator.geolocation.getCurrentPosition(function (position) {
            var startLat = position.coords.latitude;
            var startLng = position.coords.longitude;

            var data = {
                start_lat: startLat,
                start_lng: startLng,
                end_lat: destinationLat,
                end_lng: destinationLng,
                mode: mode
            };

            fetch('/find-routes', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(data)
            })
            .then(response => response.json())
            .then(routeData => {
                if (routeData.error) {
                    alert(routeData.error);
                } else {
                    displayRoute(routeData);
                }
            })
            .catch(error => console.error('Error:', error));
        });
    } else {
        alert("Géolocalisation non supportée par ce navigateur.");
    }
}

function displayRoute(routeData) {
    // Clear existing routes on the map
    if (window.currentRoute) {
        map.removeLayer(window.currentRoute);
    }

    var latlngs = [];

    if (routeData.routes.walk_to_start) {
        routeData.routes.walk_to_start.forEach(coord => {
            latlngs.push([coord[0], coord[1]]);
        });
    }

    if (routeData.routes.bike_route) {
        routeData.routes.bike_route.forEach(coord => {
            latlngs.push([coord[0], coord[1]]);
        });
    } else if (routeData.routes.public_transport_route) {
        routeData.routes.public_transport_route.forEach(coord => {
            latlngs.push([coord[0], coord[1]]);
        });
    }

    if (routeData.routes.walk_to_end) {
        routeData.routes.walk_to_end.forEach(coord => {
            latlngs.push([coord[0], coord[1]]);
        });
    }

    window.currentRoute = L.polyline(latlngs, { color: 'blue' }).addTo(map);
    map.fitBounds(window.currentRoute.getBounds());

    alert(`Itinéraire trouvé ! Mode de transport : ${routeData.transport_mode}, CO2 émissions : ${routeData.co2_emissions}g, Coins gagnés : ${routeData.coins_earned}`);
}