config = {
  "app": {
    "debug": False,
    "port": 10030
  },
  "combined_features_columns": ["NOM", "DESCRIPTIF"],
  "batch_size": 32,
  "alpha": 0.5,
  "data": {
    "lieux_visite_path": "data/Lieux de visite.xlsx",
    "feedback_file": "data/custom_verbatims.csv",
    "id_login_file": "data/id_login.csv",
    "acceslibre_file": "data/acceslibre-with-web-url.csv",
    "datatour_file": "data/Lieux de visite.xlsx",
    "models": {
      "bike_model_path": "models/generalized_model_bikes_available.pkl",
      "dock_model_path": "models/generalized_model_docks_available.pkl",
      "station_id_mapping_path": "models/station_id_mapping.pkl"
    }
  },
  "osrm": {
    "url": "http://router.project-osrm.org/route/v1/foot"
  },
  "gtfs": {
    "path": "data/atoumod/"
  },
  "station_info": {
    "url": "https://gbfs.urbansharing.com/lovelolibreservice.fr/station_information.json"
  },
  "station_status": {
    "url": "https://gbfs.urbansharing.com/lovelolibreservice.fr/station_status.json"
  },
  "overpass": {
    "query": """[out:json][timeout:25];
    (
      node["amenity"="drinking_water"](49.0, -1.7, 50.0, 1.5);
      way["amenity"="drinking_water"](49.0, -1.7, 50.0, 1.5);
      relation["amenity"="drinking_water"](49.0, -1.7, 50.0, 1.5);
    );
    out body;
    >;
    out skel qt;"""
  },
  "openagenda": {
    "api_key": "b12a402427294e3099ee3b9bdac69f85",
    "agenda_uids": [11317568, 11362982, 881924, 59233841, 6418448]
  },
  "embedding_model": {
    "name": "models/all-MiniLM-L6-v2"
  },
  "collection": {
    "database": {
      "users_file": "data/users.csv"
    },
    "badges": {
      "Incontournable de la Normandie": "A visité la Cathédrale de Rouen",
      "Connaisseur de la région": "A visité 50% de la Normandie",
      "Ami des deux roues": "S'est déplacé en vélo sur plus de 10 km"
    },
    "tiers": [
      {
        "tier": 1,
        "min_nc": 1000,
        "max_nc": 2000,
        "rewards": [
          {"name": "Carte postale offerte", "cost": 1000},
          {"name": "Réduction 10% sur toutes les stations vélos 24H", "cost": 1000},
          {"name": "Goodies mystère", "cost": 1000}
        ]
      },
      {
        "tier": 2,
        "min_nc": 2000,
        "max_nc": 3000,
        "rewards": [
          {"name": "Bon d'achat - 5€", "cost": 2000},
          {"name": "Ticket de bus offert dans Rouen", "cost": 2000},
          {"name": "Boisson gratuite dans un restaurant participant", "cost": 2000}
        ]
      },
      {
        "tier": 3,
        "min_nc": 3000,
        "max_nc": 5000,
        "rewards": [
          {"name": "Bon d'achat - 15€", "cost": 3000},
          {"name": "Piste VIP", "cost": 3000},
          {"name": "Dîner spécial", "cost": 3000}
        ]
      },
      {
        "tier": 4,
        "min_nc": 5000,
        "max_nc": None,
        "rewards": [
          {"name": "Expérience deluxe", "cost": 5000},
          {"name": "Voyage offert SNCF", "cost": 5000},
          {"name": "Cadeau premium", "cost": 5000}
        ]
      }
    ]
  } , 

"prompts" : {
  "prompts": {
    "classify": {
      "text": """Respond only in French, do not translate into English.

You are a consultant who must classify the comment delimited between the XML tags <comment>.
A comment can contain multiple verbatims; assign a theme and a sentiment to each verbatim to the following themas: <themas> {themes} </themas>

Verbatims are about locations in Normandie region.
The attribution rules to help you classify the themes are as follows: {descriptions}

When the verbatim doesn't have a theme or too generic to be classified, assign the class "Autres".

All verbatims must be found in the original comment.
Special characters must be preserved.

{format_instructions}

<comment> {comment} </comment>

Let's think step by step.
1. Break down the comment delimited between the XML tags <comment> into verbatims.
2. Assign a theme, subtheme and sentiment to each verbatim.
3. Format the output in JSON as previously seen."""
    },
    "correct_output": {
      "text": """Respond only in French, do not translate into English.
The input text delimited by XML tags <input> is an output from an LLM, but unfortunately, it did not perform its job well and there are some errors.
Reformat the output according to the format_instructions.
Don't add any comments or notes.

{format_instructions}

Let's think step by step:

1. This could be a format issue, such as the addition of extra characters. If so, remove these extra characters to adhere to the format_instructions.
2. Alternatively, it could be special characters like accents that are having difficulty being encoded. If this is the case, re-encode them to a format that can handle accents, like utf-8.
example : Ă© -> é
3. If it's a different problem, you can correct it to comply with the format_instructions.
4. If the JSON is in the input, parse the JSON.
5. Just return the JSON in the format_instructions format without your comment.

<input> {input} <\\input>"""
    }
  },
  "llm": {
    "recommendation_prompt": """Tu es un LLM spécialisé dans la recommandation de sites touristiques de la région normande.
Ton but est de comprendre si l'utilisateur attend de toi une ou plusieurs recommandations touristiques.
Répond simplement par 'oui' ou 'non'. Voici la requête :

{query}""",
    "generate_response_prompt": """Voici une demande utilisateur : "{query}"
Voici la liste des sites touristiques pour répondre à cette demande. Ils ont été choisis avec soin et tu dois tous les inclure dans la réponse en précisant leur description et leur lieu :
{context}

Génère une réponse pour l'utilisateur qui récapitule tous les sites mentionnés dans la liste dans l'ordre. Peu importe si tu considères qu'ils ne sont pas pertinents, fais le récapitulatif pour tous.""",
    "no_recommendation_prompt": """Voici une demande utilisateur : "{query}"
Je n'ai trouvé aucun site touristique correspondant à cette demande. Rédige une réponse pour l'utilisateur qui explique cela clairement et propose des alternatives générales.""",
    "general_response_prompt": """Voici une demande utilisateur : "{query}"
Cette demande ne semble pas nécessiter de recommandation spécifique. Crée une réponse conversationnelle adaptée pour répondre de manière utile et engageante à cette demande."""
  }
},

"themes" : {
  "themes": [
    {
      "name": "Patrimoine historique et culturel",
      "description": (
        "Le thème 'Patrimoine historique et culturel' englobe les aspects relatifs aux richesses historiques et culturelles de la région Normandie. "
        "Cela inclut les châteaux et manoirs, véritables témoignages architecturaux du passé, ainsi que les abbayes et monastères qui racontent l'histoire religieuse de la région. "
        "Les musées et expositions permettent de découvrir des œuvres d'art et des artefacts, tandis que les sites archéologiques révèlent des vestiges des civilisations anciennes. "
        "Les monuments historiques sont des repères importants du patrimoine, et le patrimoine industriel illustre l'évolution des activités économiques locales."
      )
    },
    {
      "name": "Paysages et nature",
      "description": (
        "Le thème 'Paysages et nature' englobe les merveilles naturelles de la Normandie, offrant une diversité de paysages à admirer. "
        "Les plages et littoraux attirent les visiteurs pour leur beauté et leur tranquillité. Les parcs naturels et réserves protègent la biodiversité et offrent des espaces pour la détente et l'observation de la faune et de la flore. "
        "Les jardins et parcs, souvent hérités de grandes demeures, sont des lieux de promenade. Les falaises et formations rocheuses impressionnent par leur majesté, "
        "tandis que les forêts et sentiers de randonnée invitent à l'exploration et à la communion avec la nature."
      )
    },
    {
      "name": "Activités et loisirs",
      "description": (
        "Le thème 'Activités et loisirs' englobe les diverses possibilités de divertissement et d'activités de plein air en Normandie. "
        "Les randonnées pédestres et cyclistes permettent de découvrir la région à son rythme. Les activités nautiques, telles que la voile et le kayak, offrent des aventures sur les eaux. "
        "Les sports équestres raviront les amateurs de chevaux, tandis que le golf propose des parcours pour les passionnés. La pêche est une activité traditionnelle, "
        "et des activités pour enfants garantissent des moments de plaisir en famille."
      )
    },
    {
      "name": "Événements et festivals",
      "description": (
        "Le thème 'Événements et festivals' englobe les nombreuses manifestations culturelles et festives de la région. "
        "Les festivals de musique animent les soirées d'été, tandis que les fêtes locales et traditionnelles célèbrent les coutumes et l'histoire. "
        "Les expositions et foires permettent de découvrir des produits locaux et des œuvres artistiques. Les spectacles et concerts offrent des moments de divertissement, "
        "et les reconstitutions historiques plongent les visiteurs dans le passé."
      )
    },
    {
      "name": "Gastronomie et produits locaux",
      "description": (
        "Le thème 'Gastronomie et produits locaux' englobe la richesse culinaire de la Normandie. "
        "Les restaurants et bistrots proposent des plats savoureux mettant en avant les produits du terroir, tels que les fromages et le cidre. "
        "Les marchés locaux sont des lieux de rencontre et de découverte des saveurs régionales. Les dégustations et visites de producteurs permettent d'apprécier la qualité des produits locaux, "
        "et les recettes locales transmettent un savoir-faire culinaire unique."
      )
    },
    {
      "name": "Hébergement et services",
      "description": (
        "Le thème 'Hébergement et services' englobe les différentes options d'hébergement et les services touristiques disponibles en Normandie. "
        "Les hôtels et auberges offrent des séjours confortables, tandis que les chambres d'hôtes et gîtes permettent de vivre une expérience plus authentique. "
        "Les campings et aires de camping-car sont idéaux pour les amateurs de plein air. Les services touristiques, tels que les guides et les visites organisées, facilitent la découverte de la région. "
        "L'accessibilité et les services pour personnes à mobilité réduite sont également pris en compte."
      )
    },
    {
      "name": "Transport et accessibilité",
      "description": (
        "Le thème 'Transport et accessibilité' englobe les moyens d'accès et de déplacement en Normandie. "
        "L'accès et le transport, incluant les routes, gares et aéroports, sont essentiels pour la mobilité des visiteurs. "
        "Les transports publics et navettes facilitent les déplacements locaux. La location de vélos et voitures offre une flexibilité supplémentaire. "
        "La signalisation et les panneaux d'information sont importants pour orienter les visiteurs et rendre leur séjour plus agréable."
      )
    },
    {
      "name": "Accueil et information",
      "description": (
        "Le thème 'Accueil et information' englobe les aspects liés à l'accueil des visiteurs et la qualité des informations fournies. "
        "Les bureaux d'information touristique sont des points de référence pour obtenir des conseils et des brochures. "
        "L'accueil et l'amabilité du personnel jouent un rôle crucial dans l'expérience des visiteurs. "
        "La disponibilité et la qualité des informations, ainsi que les brochures et cartes, sont essentielles pour une visite réussie."
      )
    },
    {
      "name": "Conservation et entretien",
      "description": (
        "Le thème 'Conservation et entretien' englobe les efforts de préservation et d'entretien des sites touristiques en Normandie. "
        "L'état de conservation des sites est un indicateur de leur soin, tandis que la propreté et l'entretien général contribuent à l'agrément des visites. "
        "La signalétique et les informations sur place sont nécessaires pour une bonne compréhension des lieux. "
        "Les mesures de préservation environnementale montrent l'engagement envers la protection de la nature."
      )
    },
    {
      "name": "Expérience utilisateur et impressions générales",
      "description": (
        "Le thème 'Expérience utilisateur et impressions générales' englobe l'évaluation globale de la visite en Normandie. "
        "La qualité globale de l'expérience est un critère clé, tout comme le rapport qualité/prix. "
        "Les recommandations et suggestions des visiteurs peuvent offrir des perspectives intéressantes, et la comparaison avec d'autres régions ou sites permet de situer la Normandie dans un contexte plus large. "
        "Les impressions esthétiques et émotionnelles des visiteurs sont également importantes pour comprendre leur ressenti."
      )
    },
    {
      "name": "Accessibilité numérique",
      "description": (
        "Le thème 'Accessibilité numérique' englobe les aspects liés à l'utilisation des outils numériques pour découvrir la Normandie. "
        "La facilité d'utilisation du site web ou de l'application est cruciale pour attirer les visiteurs. "
        "La disponibilité et la qualité des informations en ligne, ainsi que la possibilité de réservation en ligne et de billetterie, facilitent la planification des séjours. "
        "L'accessibilité pour les non-francophones est également un point important pour accueillir un public international."
      )
    },
    {
      "name": "Autres",
      "description": (
        "Le thème 'Autres' englobe tous les verbatims qui sont trop génériques ou qui traitent d'un sujet autre que les thèmes prévus."
      )
    }
  ]
}
}