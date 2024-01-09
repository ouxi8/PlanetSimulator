import matplotlib.pyplot as plt
import numpy as np


class Planeta:
    AU = 149.6e6 * 1000  # Unidades: m     Unidades astronomicas
    G = 6.67428e-11  # Constante gravitacional
    TIMESTEP = 60 * 60 * 24 * 1  # Salto de tiempo en el cual vamos a ejecutar los metodos numericos

    def __init__(self, x: float, y: float, radio: float, color: str, masa: float, nombre: str) -> None:
        """
        Constructor de la clase planeta en donde creamos un objeto de la clase Planeta
        En esta clase guardamos todos los datos relevantes a un planeta, tambine creamos los metodos necesarios para
        hacer el simulador
        :param x: Posicion x del planeta
        :param y: Posicion y del planeta
        :param radio: Radio del planeta
        :param color: Color del planeta
        :param masa: Masa del planeta
        :param nombre: Nombre del planeta
        """

        self.x = x  # Unidades: m
        self.y = y  # Unidades: m
        self.orbita = [(self.x, self.y)]  # Lista de tuplas en donde guardamos los x, y por los que ha pasado el planeta
        self.radius = radio * 1000 * 2  # Unidades: m
        self.x_vel = 0  # Unidades: m/s
        self.y_vel = 0  # Unidades: m/s
        self.color = color  # Color en el cual vamos a mostrar el planeta
        self.mass = masa  # Unidades: kg
        self.name = nombre  # Nombre del planeta

        self.es_el_sol = False  # Si el planeta es el sol
        self.distancia_al_sol = (self.x ** 2 + self.y ** 2) ** 0.5  # Distancia al sol del planeta

    def fuerza_atraccion(self, otro_planeta):
        """
        Calculamos la aceleracion ejecida por un otro planeta sobre el planeta actual
        :param otro_planeta: Objeto de la clase planeta con el cual vamos a calcular la aceleracion que ejerce sobre el planeta actual
        :return: Devuelve la aceleracion en el eje x e y
        """
        distancia_x = otro_planeta.x - self.x  # m
        distancia_y = otro_planeta.y - self.y  # m
        distancia = np.linalg.norm([distancia_x, distancia_y])

        if otro_planeta.es_el_sol:
            self.distancia_al_sol = distancia

        aceleracion_x = -self.aceleracion_x(distancia_x, distancia_y, otro_planeta.mass)
        aceleracion_y = -self.aceleracion_y(distancia_x, distancia_y, otro_planeta.mass)

        return aceleracion_x, aceleracion_y

    def euler(self, planets: list):
        """
        Metodo de euler en el cual calculamos la aceleracion neta que estan ejerciendo todos los planetas de alrededor
        sobre nosotros, a partir de ahi aplicamos el metodo para calcular la velocidad y por ello la posicion
        :param planets: Lista de todos los planetas
        """
        total_aceleracion_x = total_aceleracion_y = 0
        for planet in planets:
            if self == planet:
                continue

            aceleracion_x, aceleracion_y = self.fuerza_atraccion(planet)
            total_aceleracion_x += aceleracion_x
            total_aceleracion_y += aceleracion_y

        self.x_vel += total_aceleracion_x * self.TIMESTEP
        self.y_vel += total_aceleracion_y * self.TIMESTEP

        self.x += self.x_vel * self.TIMESTEP
        self.y += self.y_vel * self.TIMESTEP

        self.orbita.append((self.x, self.y))

    def rk4(self, masa: float) -> None:
        """
        Metodo de runge kutta en donde calcula la posicion siguiente del planeta
        :param masa: Masa del objeto al cual orbita el planeta
        """
        delta_t = Planeta.TIMESTEP
        distancia_x_al_sol = self.x
        distancia_y_al_sol = self.y

        self.distancia_al_sol = (self.x ** 2 + self.y ** 2) ** 0.5

        k0 = delta_t * self.aceleracion_x(distancia_x_al_sol, distancia_y_al_sol, masa)
        l0 = delta_t * self.aceleracion_y(distancia_x_al_sol, distancia_y_al_sol, masa)

        k1 = delta_t * self.aceleracion_x(distancia_x_al_sol + 0.5 * k0, distancia_y_al_sol + 0.5 * l0, masa)
        l1 = delta_t * self.aceleracion_y(distancia_x_al_sol + 0.5 * k0, distancia_y_al_sol + 0.5 * l0, masa)

        k2 = delta_t * self.aceleracion_x(distancia_x_al_sol + 0.5 * k1, distancia_y_al_sol + 0.5 * l1, masa)
        l2 = delta_t * self.aceleracion_y(distancia_x_al_sol + 0.5 * k1, distancia_y_al_sol + 0.5 * l1, masa)

        k3 = delta_t * self.aceleracion_x(distancia_x_al_sol + k2, distancia_y_al_sol + l2, masa)
        l3 = delta_t * self.aceleracion_y(distancia_x_al_sol + k2, distancia_y_al_sol + l2, masa)

        self.x_vel += (1 / 6) * (k0 + 2 * k1 + 2 * k2 + k3)
        self.y_vel += (1 / 6) * (l0 + 2 * l1 + 2 * l2 + l3)

        self.x += self.x_vel * delta_t
        self.y += self.y_vel * delta_t

        self.orbita.append((self.x, self.y))

    def aceleracion_x(self, x: float, y: float, masa_planeta_atrayente: float) -> float:  # Aceleracion en x
        """
        Calculo de la aceleracion en el eje x
        :param x: Posicion x del planeta atraido
        :param y: Posicion y del planeta atraido
        :param masa_planeta_atrayente: Masa del planeta atrayente
        :return: Aceleracion en el eje x
        """
        return -Planeta.G * masa_planeta_atrayente * x / ((x ** 2 + y ** 2) ** (3 / 2))

    def aceleracion_y(self, x: float, y: float, masa_planeta_atrayente: float) -> float:  # Aceleracion en y
        """
        Calculo de la aceleracion en el eje y
        :param x: Posicion x del planeta atraido
        :param y: Posicion y del planeta atraido
        :param masa_planeta_atrayente: Masa del planeta atrayente
        :return: Aceleracion en el eje y
        """
        return -Planeta.G * masa_planeta_atrayente * y / ((x ** 2 + y ** 2) ** (3 / 2))


def calculo_velocidad_circular(constante_gravitacional: float, masa: float, distancia: float) -> float:
    """
    Calculo de la velocidad centrifuga necesaria para que el planeta orbite alrededor del sol
    :param constante_gravitacional:
    :param masa:
    :param distancia:
    :return:velocidad circular
    """
    return (constante_gravitacional * masa / distancia) ** 0.5


# Example usage:
sol = Planeta(0, 0, 696340 * 25, 'orange', 3.955e30, 'Sun')
sol.es_el_sol = True
mercury = Planeta(0.387 * Planeta.AU, 0, 2489.7 * 1000, 'gray', 3.30e23, 'Mercury')
venus = Planeta(-0.723 * Planeta.AU, 0, 6051.8 * 1000, 'yellow', 4.8685e24, 'Venus')
tierra = Planeta(1 * Planeta.AU, 0, 6371 * 1000, 'blue', 5.972e24, 'Earth')
marte = Planeta(-1.524 * Planeta.AU, 0, 3389.5 * 1000, 'brown', 6.39e24, 'Mars')

mercury.y_vel = calculo_velocidad_circular(Planeta.G, sol.mass, mercury.distancia_al_sol)
venus.y_vel = -calculo_velocidad_circular(Planeta.G, sol.mass, venus.distancia_al_sol)
tierra.y_vel = calculo_velocidad_circular(Planeta.G, sol.mass, tierra.distancia_al_sol)
marte.y_vel = -calculo_velocidad_circular(Planeta.G, sol.mass, marte.distancia_al_sol)

planetas = [sol, tierra, marte, mercury, venus]

# Estos planetas no usamos las distancias y radios reales ya que son numeros muy grandes
# Las masas si son reales (datos de Wikipedia)
masa_tierra = 5.972e24

jupiter = Planeta(2.1 * Planeta.AU, 0, 69911 * 300, 'pink', 318 * masa_tierra, 'Jupiter')
saturno = Planeta(-2.9 * Planeta.AU, 0, 58232 * 300, 'violet', 95 * masa_tierra, 'Venus')
urano = Planeta(3.5 * Planeta.AU, 0, 25362 * 300, 'green', 14.6 * masa_tierra, 'Urano')
neptuno = Planeta(-4 * Planeta.AU, 0, 24622 * 300, 'cyan', 17.2 * masa_tierra, 'Neptuno')

test = Planeta(-4.5 * Planeta.AU, 0, 69911 * 300, 'black', 318 * masa_tierra, 'Test')
test1 = Planeta(-4.8 * Planeta.AU, 0, 58232 * 300, 'yellow', 95 * masa_tierra, 'test1')
test2 = Planeta(5.3 * Planeta.AU, 0, 25362 * 300, 'darkblue', 14.6 * masa_tierra, 'test2')
test3 = Planeta(-5.8 * Planeta.AU, 0, 24622 * 300, 'beige', 17.2 * masa_tierra, 'test3')

test.y_vel = calculo_velocidad_circular(Planeta.G, sol.mass, jupiter.distancia_al_sol)
test1.y_vel = -calculo_velocidad_circular(Planeta.G, sol.mass, saturno.distancia_al_sol)
test2.y_vel = calculo_velocidad_circular(Planeta.G, sol.mass, urano.distancia_al_sol)
test3.y_vel = -calculo_velocidad_circular(Planeta.G, sol.mass, neptuno.distancia_al_sol)

jupiter.y_vel = calculo_velocidad_circular(Planeta.G, sol.mass, jupiter.distancia_al_sol)
saturno.y_vel = -calculo_velocidad_circular(Planeta.G, sol.mass, saturno.distancia_al_sol)
urano.y_vel = calculo_velocidad_circular(Planeta.G, sol.mass, urano.distancia_al_sol)
neptuno.y_vel = -calculo_velocidad_circular(Planeta.G, sol.mass, neptuno.distancia_al_sol)

planetas.append(jupiter)
planetas.append(saturno)
planetas.append(urano)
planetas.append(neptuno)

planetas.append(test)
planetas.append(test1)
planetas.append(test2)
planetas.append(test3)

tamano_ejes = 6
# Create a Matplotlib plot
plt.figure(figsize=(10, 10))
ax = plt.gca()

for i in range(5000):
    ax.cla()  # Limpiamos la grafica

    esquina_x = -8.5 * Planeta.AU
    esquina_y = 5.5 * Planeta.AU

    for planeta in planetas:
        # Pintamos todos los planetas
        ax.add_patch(
            plt.Circle((planeta.x, planeta.y), planeta.radius, color=planeta.color, fill=True, label=planeta.name))

        # Si el planeta no es el sol pintamos las orbitas, suponemos que el sol no se mueve
        if not planeta.es_el_sol:
            orbit_x, orbit_y = zip(*planeta.orbita)
            ax.plot(orbit_x, orbit_y, color=planeta.color, linewidth=1)

            # Calculamos la posicion siguiente del planeta, ya sea por euler o runge kutta

            # Por runge kutta
            planeta.rk4(sol.mass)

            # Por euler
            # planeta.euler(planetas)

            # Añadimos etiquetas en la esquina superior izquierda en donde muestre la distancia de cada planeta al sol
            distance_text = f"{planeta.name}: {planeta.distancia_al_sol:.5e} m"
            ax.text(esquina_x, esquina_y, distance_text, ha='left', va='bottom', color=planeta.color)
            esquina_y -= 0.3 * Planeta.AU  # Bajamos la siguiente etiqueta para que no se superpongan

    ax.set_aspect('equal')  # Set the aspect ratio to 'equal'
    ax.set_xlim(-tamano_ejes * Planeta.AU, tamano_ejes * Planeta.AU)  # Limites del eje x
    ax.set_ylim(-tamano_ejes * Planeta.AU, tamano_ejes * Planeta.AU)  # Limites del eje y
    ax.axis('off')  # Activamos los ejes
    ax.legend()  # Activamos la leyenda

    plt.draw()
    plt.pause(0.05)  # Tiempo de pausa

plt.show()
