# well_map_builder.py
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import gaussian_filter
from pathlib import Path
from Irap import Irap

def generate_well_map(params: dict | None = None) -> dict:
    """
    Строит интерактивную карту скважин (траектории, точки среза, зоны дренирования).
    Параметры:
        dataset_path, inclinometry_path, ... – пути к исходным данным (оставлены значения по умолчанию)
        output_html – имя выходного HTML-файла (сохраняется в папке output или рядом)
    Возвращает:
        {"status": "ok", "data": {"html_path": str}, "error": None}
        или {"status": "error", ...}
    """
    if params is None:
        params = {}

    dataset_path = params.get("dataset_path", r"Датасет_обновл.xlsx")
    inclinometry_path = params.get("inclinometry_path", r"Инклинометрия.xlsx")
    constructed_wells_file = params.get("constructed_wells_file", r"constructed_wells_2.xlsx")
    slice_points_file = params.get("slice_points_file", r"slice_points.xlsx")
    kpngz_file = params.get("kpngz_file", r"КПНГЗ_2000.grd")
    kr_file = params.get("kr_file", r"АО.кр.кол_2000.grd")
    pod_file = params.get("pod_file", r"АО.под.кол_2000.grd")
    output_html = params.get("output_html", "well_map.html")

    try:
        output_dir = Path(__file__).resolve().parent / "output"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / Path(output_html).name

        # Загрузка данных из Excel
        df_dataset = pd.read_excel(dataset_path)
        df_dataset["СКВАЖИНА"] = df_dataset["СКВАЖИНА"].astype(str).str.strip()

        df_inclin = pd.read_excel(inclinometry_path)
        df_inclin["СКВАЖИНА"] = df_inclin["СКВАЖИНА"].astype(str).str.strip()

        common_wells = set(df_dataset["СКВАЖИНА"]) & set(df_inclin["СКВАЖИНА"])

        df_inclin_filtered = df_inclin[df_inclin["СКВАЖИНА"].isin(common_wells)].copy()
        df_inclin_filtered = df_inclin_filtered[["СКВАЖИНА", "ГЛУБИНА", "УГОЛ", "АЗИМУТ", "АБС_ГЛУБИНА"]]

        # Загрузка карт поверхностей
        kpngz = Irap(kpngz_file).load()
        kr = Irap(kr_file).load()
        pod = Irap(pod_file).load()

        fig = go.Figure()
        contour_trace = go.Contour(
            z=kpngz.value,
            x=kpngz.x,
            y=kpngz.y,
            colorscale="cividis",
            colorbar=dict(title="КПНГЗ", len=0.5, x=0.2, orientation="h"),
            contours=dict(showlines=False),
            name="КПНГЗ",
            connectgaps=True
        )
        fig.add_trace(contour_trace)

        if kr.x.ndim > 1:
            kr_x = kr.x[0, :]
            kr_y = kr.y[:, 0]
        else:
            kr_x = kr.x
            kr_y = kr.y
        kr_smooth = gaussian_filter(-kr.value, sigma=2.0)
        kr_interp = RegularGridInterpolator((kr_y, kr_x), kr_smooth, method='linear',
                                            bounds_error=False, fill_value=np.nan)

        X, Y = np.meshgrid(kpngz.x, kpngz.y)

        df_slice_points = pd.read_excel(slice_points_file)
        mouth_points = []

        # --- Обработка скважин из Инклинометрии ---
        for well in sorted(common_wells):
            df_well = df_inclin_filtered[df_inclin_filtered["СКВАЖИНА"] == well].copy()
            df_well.sort_values(by="ГЛУБИНА", ascending=False, inplace=True)
            df_well.reset_index(drop=True, inplace=True)

            well_data = df_dataset[df_dataset["СКВАЖИНА"] == well]
            if well_data.empty:
                continue

            x_end, y_end = well_data.iloc[0][["Y_T3", "X_T_3"]]
            if pd.isna(x_end) or pd.isna(y_end):
                print(f"Скважина {well}: начальные координаты забоя (x_end, y_end) содержат NaN. Пропускаем.")
                continue

            if df_well.shape[0] < 2:
                print(f"Скважина {well}: недостаточно точек для построения траектории. Пропускаем.")
                continue

            for col in ["ГЛУБИНА", "УГОЛ", "АЗИМУТ", "АБС_ГЛУБИНА"]:
                df_well[col] = pd.to_numeric(df_well[col], errors="coerce")

            x_coords = [x_end]
            y_coords = [y_end]
            z_coords = [-df_well["АБС_ГЛУБИНА"].values[0]]

            for i in range(1, len(df_well)):
                delta_d = df_well.loc[i - 1, "ГЛУБИНА"] - df_well.loc[i, "ГЛУБИНА"]
                angle_rad = np.deg2rad(df_well.loc[i, "УГОЛ"])
                azimuth_rad = np.deg2rad(df_well.loc[i, "АЗИМУТ"])
                horizontal_disp = delta_d * np.sin(angle_rad)
                dx = horizontal_disp * np.sin(azimuth_rad)
                dy = horizontal_disp * np.cos(azimuth_rad)
                x_coords.append(x_coords[-1] - dx)
                y_coords.append(y_coords[-1] - dy)
                z_coords.append(-df_well.loc[i, "АБС_ГЛУБИНА"])

            mouth_x = x_coords[-1]
            mouth_y = y_coords[-1]

            if pd.isna(mouth_x) or pd.isna(mouth_y):
                found_valid_point = False
                for i in range(len(x_coords) - 2, -1, -1):
                    if not (pd.isna(x_coords[i]) or pd.isna(y_coords[i])):
                        mouth_x = x_coords[i]
                        mouth_y = y_coords[i]
                        found_valid_point = True
                        break
                if not found_valid_point:
                    continue

            mouth_points.append({
                "well": "5"+well+"X",
                "x_mouth": mouth_x,
                "y_mouth": mouth_y
            })
            print(f"Точка устья для скважины {well}: X={mouth_x:.2f}, Y={mouth_y:.2f}")

            slice_match = df_slice_points[df_slice_points["СКВАЖИНА"] == well]
            if not slice_match.empty:
                x_slice = slice_match.iloc[0]["x_slice"]
                y_slice = slice_match.iloc[0]["y_slice"]
                slice_distance = np.sqrt((x_slice - mouth_x)**2 + (y_slice - mouth_y)**2)
                fig.add_trace(go.Scatter(x=[x_slice], y=[y_slice], mode="markers", marker=dict(size=6, color="green", symbol="x"), name=f"Точка среза {well}"))

                # Найти предыдущую точку
                min_dist = float('inf')
                prev_x, prev_y = None, None
                for i in range(len(x_coords)):
                    dist_to_slice = np.sqrt((x_coords[i] - x_slice)**2 + (y_coords[i] - y_slice)**2)
                    if dist_to_slice < min_dist:
                        min_dist = dist_to_slice
                        if i > 0:
                            prev_x, prev_y = x_coords[i-1], y_coords[i-1]
                if prev_x is not None and prev_y is not None:
                    base_azimuth = np.arctan2(prev_x - x_slice, prev_y - y_slice)
                    x = slice_distance
                    sigma = abs(-9e-8 * x**3 + 0.0003 * x**2 - 0.4604 * x + 336.5)
                    if sigma > 180:
                        sigma = 180
                    sigma_rad = np.deg2rad(sigma)
                    line_length = 700
                    azimuth_1 = base_azimuth + sigma_rad
                    azimuth_2 = base_azimuth - sigma_rad
                    x_line1_end = x_slice + line_length * np.sin(azimuth_1)
                    y_line1_end = y_slice + line_length * np.cos(azimuth_1)
                    x_line2_end = x_slice + line_length * np.sin(azimuth_2)
                    y_line2_end = y_slice + line_length * np.cos(azimuth_2)

                    fig.add_trace(go.Scatter(x=[x_slice, x_line1_end], y=[y_slice, y_line1_end], mode="lines", line=dict(width=1, color="cyan"), name=f"{well}"))
                    fig.add_trace(go.Scatter(x=[x_slice, x_line2_end], y=[y_slice, y_line2_end], mode="lines", line=dict(width=1, color="cyan"), name=f"{well}"))

                    arc_radius = 700
                    A = np.array([x_line1_end, y_line1_end])
                    B = np.array([x_line2_end, y_line2_end])
                    chord_length = np.linalg.norm(B - A)
                    if arc_radius < chord_length/2:
                        arc_radius = chord_length/2 + 1e-6
                    if np.isclose(sigma, 180, atol=1e-6):
                        circle_angles = np.linspace(0, 2*np.pi, 100)
                        circle_x = x_slice + line_length * np.cos(circle_angles)
                        circle_y = y_slice + line_length * np.sin(circle_angles)
                        fig.add_trace(go.Scatter(x=circle_x, y=circle_y, mode="lines", line=dict(width=1, color="orange"), name=f"{well}"))
                        fig.add_trace(go.Scatter(x=circle_x, y=circle_y, mode="lines", fill="toself", fillcolor="rgba(0,255,255,0.1)", line=dict(width=0), name=f"{well}"))
                    else:
                        M = (A + B)/2
                        d = np.sqrt(arc_radius**2 - (chord_length/2)**2)
                        v = B - A
                        v_norm = v / np.linalg.norm(v)
                        perp = np.array([-v_norm[1], v_norm[0]])
                        if sigma > 90:
                            center = M - d * perp
                        else:
                            center = M + d * perp
                        angle_A = np.arctan2(A[1]-center[1], A[0]-center[0])
                        angle_B = np.arctan2(B[1]-center[1], B[0]-center[0])
                        if angle_B < angle_A:
                            angle_B += 2*np.pi
                        arc_angles = np.linspace(angle_A, angle_B, 50)
                        arc_x = center[0] + arc_radius * np.cos(arc_angles)
                        arc_y = center[1] + arc_radius * np.sin(arc_angles)
                        fig.add_trace(go.Scatter(x=arc_x, y=arc_y, mode="lines", line=dict(width=1, color="orange"), name=f"{well}"))
                        sector_x = [x_slice, x_line1_end] + list(arc_x) + [x_line2_end, x_slice]
                        sector_y = [y_slice, y_line1_end] + list(arc_y) + [y_line2_end, y_slice]
                        fig.add_trace(go.Scatter(x=sector_x, y=sector_y, mode="lines", fill="toself", fillcolor="rgba(0,255,255,0.1)", line=dict(width=0), name=f"{well}"))

            kr_vals = [kr_interp((yi, xi)) for xi, yi in zip(x_coords, y_coords)]
            kr_vals = np.array(kr_vals)

            crossing_index = None
            for i in range(1, len(z_coords)):
                if z_coords[i] >= kr_vals[i] and z_coords[i-1] < kr_vals[i-1]:
                    crossing_index = i
                    break

            if crossing_index is None:
                fig.add_trace(go.Scatter(x=x_coords, y=y_coords, mode="lines", line=dict(width=1, color="red"), name=f"Скважина {well} (без пересечения)"))
            else:
                fig.add_trace(go.Scatter(x=x_coords[:crossing_index+1], y=y_coords[:crossing_index+1], mode="lines", line=dict(width=1, color="blue"), name=f"Скважина {well} (в пласте)"))
                fig.add_trace(go.Scatter(x=x_coords[crossing_index:], y=y_coords[crossing_index:], mode="lines", line=dict(width=1, color="orange"), name=f"Скважина {well} (над пластом)"))
                fig.add_trace(go.Scatter(x=[x_coords[crossing_index]], y=[y_coords[crossing_index]], mode="markers", marker=dict(size=3, color="pink"), name=f"Пересечение {well}"))

                # Зона дренирования (эллипс)
                Mx = np.mean(x_coords[:crossing_index+1])
                My = np.mean(y_coords[:crossing_index+1])
                Qx, Qy = x_end, y_end
                Px, Py = x_coords[crossing_index], y_coords[crossing_index]
                d_heel = np.sqrt((Mx - Qx)**2 + (My - Qy)**2)
                d_cross = np.sqrt((Mx - Px)**2 + (My - Py)**2)
                a = max(d_heel, d_cross) + 250
                b = 250
                angle = np.arctan2(Py - Qy, Px - Qx)
                theta = np.linspace(0, 2*np.pi, 100)
                ellipse_x = Mx + a*np.cos(theta)*np.cos(angle) - b*np.sin(theta)*np.sin(angle)
                ellipse_y = My + a*np.cos(theta)*np.sin(angle) + b*np.sin(theta)*np.cos(angle)
                mask = (((X - Mx)*np.cos(angle) + (Y - My)*np.sin(angle))**2 / a**2 +
                        ((X - Mx)*np.sin(angle) - (Y - My)*np.cos(angle))**2 / b**2) <= 1
                kpngz.value[mask] = 0.1

        # Добавление устьев
        mouth_x_coords = [p["x_mouth"] for p in mouth_points]
        mouth_y_coords = [p["y_mouth"] for p in mouth_points]
        mouth_names = [p['well'] for p in mouth_points]
        fig.add_trace(go.Scatter(x=mouth_x_coords, y=mouth_y_coords, mode="markers+text", marker=dict(size=4, color="white", symbol="diamond"), text=mouth_names, textfont=dict(color="rgba(255,255,255,0.5)"), textposition="top center", name="Устья скважин"))

        # Обработка скважин ЗБС
        df_constructed = pd.read_excel(constructed_wells_file)
        for idx, well in df_constructed.iterrows():
            x_start, y_start = well["x_start"], well["y_start"]
            x_end, y_end = well["x_end"], well["y_end"]
            well_name = well.get("name", f"ЗБС_{idx}")
            fig.add_trace(go.Scatter(x=[x_start, x_end], y=[y_start, y_end], mode="lines", line=dict(width=2, color="black"), name=f"Скважина ЗБС {well_name}"))
            Mx = (x_start + x_end)/2
            My = (y_start + y_end)/2
            Qx, Qy = x_end, y_end
            Px, Py = x_start, y_start
            d_heel = np.sqrt((Mx - Qx)**2 + (My - Qy)**2)
            d_cross = np.sqrt((Mx - Px)**2 + (My - Py)**2)
            a = max(d_heel, d_cross) + 250
            b = 250
            angle = np.arctan2(Py - Qy, Px - Qx)
            theta = np.linspace(0, 2*np.pi, 100)
            ellipse_x = Mx + a*np.cos(theta)*np.cos(angle) - b*np.sin(theta)*np.sin(angle)
            ellipse_y = My + a*np.cos(theta)*np.sin(angle) + b*np.sin(theta)*np.cos(angle)
            fig.add_trace(go.Scatter(x=ellipse_x, y=ellipse_y, mode="lines", line=dict(width=1, color="magenta", dash="dot"), name=f"Зона дренирования ЗБС {well_name}"))

        # Обновляем контур
        fig.data[0].z = kpngz.value

        fig.update_layout(
            title=dict(text="2D модель с линиями"),
            xaxis=dict(title="X", scaleanchor="y", automargin=True),
            yaxis=dict(title="Y", scaleanchor="x", automargin=True),
            showlegend=True
        )

        # Сохраняем HTML
        fig.write_html(str(output_path))

        return {"status": "ok", "data": {"html_path": str(output_path)}, "error": None}

    except Exception as e:
        return {"status": "error", "data": None, "error": str(e)}
