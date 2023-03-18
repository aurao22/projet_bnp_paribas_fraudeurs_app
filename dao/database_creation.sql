-- ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
-- DATABASE INITIALISATION SCRIPT
-- ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
drop database if exists bnp;
create database bnp;
use bnp;

-- ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
-- TABLE CREATION
-- ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
DROP TABLE IF EXISTS `bnp`.`catalogue`;
DROP TABLE IF EXISTS `bnp`.`fabricant`;
DROP TABLE IF EXISTS `bnp`.`categorie`;

CREATE TABLE IF NOT EXISTS `bnp`.`categorie` (
  `id_categorie` INT NOT NULL AUTO_INCREMENT,
  `libelle` VARCHAR(100) NULL,
  PRIMARY KEY (`id_categorie`))
ENGINE = InnoDB;


CREATE TABLE IF NOT EXISTS `bnp`.`fabricant` (
  `id` INT NOT NULL AUTO_INCREMENT,
  `designation` VARCHAR(100) NOT NULL,
  `contact` VARCHAR(100) NULL,
  PRIMARY KEY (`id`))
ENGINE = InnoDB;


CREATE TABLE IF NOT EXISTS `bnp`.`catalogue` (
  `code_produit` VARCHAR(100) NOT NULL,
  `modele` TEXT NOT NULL,
  `PU` REAL NULL,
  `fabricant` INT,
  `categorie` INT,
  PRIMARY KEY (`code_produit`),
  FOREIGN KEY (`fabricant`) REFERENCES `bnp`.`fabricant` (`id`),
  FOREIGN KEY (`categorie`)  REFERENCES `bnp`.`categorie` (`id_categorie`)
)
ENGINE = InnoDB;


SELECT code_produit, modele, PU, libelle, designation FROM catalogue, categorie, fabricant WHERE catalogue.categorie = categorie.id_categorie AND catalogue.fabricant = fabricant.id LIMIT 100;